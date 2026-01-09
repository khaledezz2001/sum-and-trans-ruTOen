FROM runpod/pytorch:2.1.0-py3.10-cuda11.8

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# System deps (only what OCR really needs)
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Python deps
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# 🔥 PRE-DOWNLOAD GOT-OCR MODEL (VERY IMPORTANT)
# -------------------------------------------------
ENV HF_HOME=/models/hf

RUN python - <<'EOF'
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

print("Downloading GOT-OCR2_0 files...")
AutoProcessor.from_pretrained(MODEL_ID)
AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)
print("GOT-OCR2_0 downloaded")
EOF

# -------------------------------------------------
# App
# -------------------------------------------------
COPY handler.py .

CMD ["python", "-u", "handler.py"]
