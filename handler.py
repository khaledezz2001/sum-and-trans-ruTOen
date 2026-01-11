import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PAGES = 20

processor = None
model = None


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img.thumbnail((2048, 2048), Image.BICUBIC)
    return img


def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(pdf_bytes, dpi=200)
    return images[:MAX_PAGES]


# ===============================
# LOAD MODEL ONCE
# ===============================
def load_model():
    global processor, model
    if model is not None:
        return

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    log("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=True
    )

    model.eval()
    log("RolmOCR model loaded")


# ===============================
# OCR ONE PAGE
# ===============================
def ocr_page(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract all readable text from this page."}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1200,
            temperature=0.1
        )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1]

    decoded = decoded.strip()
    while decoded.startswith(("system\n", "user\n", ".")):
        decoded = decoded.split("\n", 1)[-1].strip()

    return decoded


# ===============================
# HANDLER
# ===============================
def handler(event):
    load_model()
    PREFIX = "Extract all readable text from this page.\nassistant\n"
    pages = []

    if "image" in event["input"]:
        pages = [decode_image(event["input"]["image"])]

    elif "file" in event["input"]:
        pages = decode_pdf(event["input"]["file"])

    else:
        return {
            "status": "error",
            "message": "Missing image or file"
        }

    extracted_pages = []
    full_text = []

    for i, page in enumerate(pages, start=1):
        text = ocr_page(page)
        extracted_pages.append({
            "page": i,
            "text": text.replace(PREFIX, "", 1)
        })
        full_text.append(text.replace(PREFIX, "", 1))

    return {
        "status": "success",
        "total_pages": len(extracted_pages),
        "extracted_text": "\n\n".join(full_text),
        "pages": extracted_pages
    }


# ===============================
# PRELOAD
# ===============================
log("Preloading model...")
load_model()

runpod.serverless.start({
    "handler": handler
})
