import os
import base64
import io
import re
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


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img.thumbnail((2048, 2048), Image.BICUBIC)
    return img


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
    log("Model loaded successfully")


# ===============================
# PARTY TABLE NORMALIZER
# ===============================
FIELDS = ["Наименование", "ИНН", "ОГРН", "Адрес", "Представитель"]


def extract_party(block: str) -> dict:
    data = {}
    for field in FIELDS:
        m = re.search(rf"{field}:\s*(.+)", block)
        if m:
            data[field] = m.group(1).strip()
    return data


def party_table(title: str, data: dict) -> str:
    table = f"### {title}\n\n"
    table += "| Поле | Значение |\n|------|----------|\n"
    for f in FIELDS:
        table += f"| {f} | {data.get(f, '')} |\n"
    return table


def normalize_parties(text: str) -> str:
    # Extract raw blocks
    exec_block = re.search(r"Исполнитель:(.+?)(Заказчик:)", text, re.S)
    cust_block = re.search(r"Заказчик:(.+?)(\n\n|$)", text, re.S)

    if not exec_block or not cust_block:
        return text  # fallback

    exec_data = extract_party(exec_block.group(1))
    cust_data = extract_party(cust_block.group(1))

    tables = (
        party_table("Исполнитель", exec_data)
        + "\n"
        + party_table("Заказчик", cust_data)
    )

    # Replace entire section
    text = re.sub(
        r"### 1\. СТОРОНЫ ДОГОВОРА.+?### 2\.",
        "### 1. СТОРОНЫ ДОГОВОРА\n\n"
        + tables
        + "\n### 2.",
        text,
        flags=re.S,
    )

    return text


# ===============================
# HANDLER
# ===============================
def handler(event):
    log("Handler called")
    load_model()

    image = decode_image(event["input"]["image"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Extract all text from this document.\n\n"
                        "Rules:\n"
                        "1. Preserve headings and numbering.\n"
                        "2. For each party, list fields on separate lines:\n"
                        "   Наименование:\n"
                        "   ИНН:\n"
                        "   ОГРН:\n"
                        "   Адрес:\n"
                        "   Представитель:\n"
                        "3. Do NOT format parties as tables.\n"
                        "4. Normalize paragraphs.\n"
                        "5. Keep Russian language.\n"
                        "6. At the end, state whether signatures and stamps are present.\n"
                        "7. Output Markdown ONLY."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1600,
            temperature=0.1
        )

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    # ---- CLEAN CHAT WRAPPER ----
    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1]

    decoded = decoded.strip()
    while decoded.startswith(("system\n", "user\n", ".")):
        decoded = decoded.split("\n", 1)[-1].strip()

    # ---- NORMALIZE PARTIES ----
    decoded = normalize_parties(decoded)

    return {
        "format": "markdown",
        "text": decoded
    }


# ===============================
# PRELOAD
# ===============================
log("Preloading model...")
load_model()

runpod.serverless.start({"handler": handler})
