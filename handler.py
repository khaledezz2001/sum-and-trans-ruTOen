import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PAGES = 20

processor = None
model = None


# ===============================
# LOAD MODEL
# ===============================
def load_model():
    global processor, model
    if model is not None:
        return

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=True
    )

    model.eval()


# ===============================
# INPUT DECODERS
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img.thumbnail((2048, 2048), Image.BICUBIC)
    return img


def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    return convert_from_bytes(pdf_bytes, dpi=200)[:MAX_PAGES]


# ===============================
# OCR (RU)
# ===============================
def ocr_page(image):
    prompt = "Extract all readable text from this page in the original language."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1200
        )

    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()


# ===============================
# TRANSLATE RU → EN
# ===============================
def translate_to_english(text_ru):
    prompt = (
        "Translate the following text from Russian to clear, professional English.\n\n"
        "Do not summarize. Do not explain. Translate faithfully.\n\n"
        f"{text_ru}"
    )

    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1600
        )

    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()


# ===============================
# HANDLER
# ===============================
def handler(event):
    load_model()

    if "image" in event["input"]:
        pages = [decode_image(event["input"]["image"])]
    elif "file" in event["input"]:
        pages = decode_pdf(event["input"]["file"])
    else:
        return {"status": "error", "message": "Missing image or file"}

    results = []
    all_ru = []
    all_en = []

    for i, page in enumerate(pages, 1):
        ru = ocr_page(page)
        en = translate_to_english(ru)

        results.append({
            "page": i,
            "text_ru": ru,
            "text_en": en
        })

        all_ru.append(ru)
        all_en.append(en)

    return {
        "status": "success",
        "total_pages": len(results),
        "extracted_text_ru": "\n\n".join(all_ru),
        "extracted_text_en": "\n\n".join(all_en),
        "pages": results
    }


# ===============================
# START
# ===============================
load_model()
runpod.serverless.start({"handler": handler})
