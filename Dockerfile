import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

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

    # ===============================
    # QWEN2.5-VL CHAT FORMAT (REQUIRED)
    # ===============================
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
                        "1. Preserve headings and numbering exactly.\n"
                        "2. For EACH party, output a Markdown table with EXACTLY two columns:\n"
                        "   | Поле | Значение |\n"
                        "   Use ONLY these fields and this order:\n"
                        "   Наименование, ИНН, ОГРН, Адрес, Представитель.\n"
                        "3. Do NOT merge fields into paragraphs.\n"
                        "4. Normalize paragraphs (no OCR line noise).\n"
                        "5. Keep original language (Russian).\n"
                        "6. At the end, explicitly state whether signatures and stamps are present.\n"
                        "7. Do NOT add explanations, comments, or metadata.\n"
                        "8. Output Markdown ONLY."
                    )
                }
            ]
        }
    ]

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

    # ===============================
    # HARD OUTPUT CLEANUP
    # ===============================
    # Remove everything before assistant response
    for marker in ["assistant\n", "assistant\r\n"]:
        if marker in decoded:
            decoded = decoded.split(marker, 1)[1]

    decoded = decoded.strip()

    # Remove any leftover system/user noise
    while decoded.startswith(("system\n", "user\n", ".")):
        decoded = decoded.split("\n", 1)[-1].strip()

    return {
        "format": "markdown",
        "text": decoded
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
