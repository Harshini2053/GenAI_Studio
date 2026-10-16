# GenAI Studio – text-to-image + img2img demo (FastAPI + Stable Diffusion)

## Run (2 minutes)
```bash
pip install -r requirements.txt
python app.py                 # real model (SD-Turbo, downloads ~2.5 GB once)
# UI only, no GPU/model needed:
MOCK=1 python app.py          # Windows PowerShell: $env:MOCK=1; python app.py
```
Open http://localhost:8000

## Options (env vars)
- `MODEL_ID`  default `stabilityai/sd-turbo` (any SD/SDXL-turbo style Hugging Face model)
- `LORA_PATH` path or HF repo of a LoRA, e.g. your own fine-tuned style
- `MOCK=1`    placeholder images, instant, no downloads

## Docker (GPU)
```bash
docker build -t genai-studio .
docker run --gpus all -p 8000:8000 genai-studio
```

## What it shows
Text-to-image, img2img, LoRA loading, seed control, fp16 GPU inference,
FastAPI model serving, Docker packaging, latency reporting.
