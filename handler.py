import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===============================
# OFFLINE MODE (RUNTIME ONLY)
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

processor = None
model = None


def log(msg: str):
    print(f"[BOOT] {msg}", flush=True)


def decode_image(b64: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    image.thumbnail((2048, 2048), Image.BICUBIC)
    return image


# ===============================
# LOAD MODEL ONCE
# ===============================
def load_model():
    global processor, model
    if model is not None:
        return

    log("Loading RolmOCR processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    log("Loading RolmOCR model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=True
    )

    model.eval()
    log("RolmOCR model loaded successfully")


# ===============================
# HANDLER
# ===============================
def handler(event):
    log("Handler called")
    load_model()

    if "image" not in event["input"]:
        return {"error": "Missing image"}

    image = decode_image(event["input"]["image"])

    # ✅ CORRECT QWEN2.5-VL MESSAGE FORMAT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Extract all text from this document and return a CLEAN, STRUCTURED "
                        "Markdown representation.\n\n"
                        "Rules:\n"
                        "1. Preserve headings and numbering.\n"
                        "2. Extract parties into Markdown tables with fields: "
                        "Наименование, ИНН, ОГРН, Адрес, Представитель.\n"
                        "3. Normalize paragraphs (no OCR line noise).\n"
                        "4. Keep original language (Russian).\n"
                        "5. At the end, explicitly state whether signatures and stamps are present.\n"
                        "6. Do NOT add explanations, comments, or metadata.\n"
                        "7. Output Markdown ONLY."
                    )
                }
            ]
        }
    ]

    # ✅ Inject image tokens via chat template
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1400,
            temperature=0.1
        )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    # ✅ Strip chat wrapper
    if "assistant" in decoded:
        result = decoded.split("assistant", 1)[-1].strip()
    else:
        result = decoded.strip()

    return {
        "format": "markdown",
        "text": result
    }


# ===============================
# PRELOAD AT STARTUP
# ===============================
log("Preloading model at startup...")
load_model()

# ===============================
# START RUNPOD
# ===============================
runpod.serverless.start({
    "handler": handler
})
