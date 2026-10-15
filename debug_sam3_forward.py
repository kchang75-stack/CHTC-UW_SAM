#!/usr/bin/env python3
import os
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

token_path = "hf_token.txt"
hf_token = None

if os.path.exists(token_path):
    with open(token_path, "r") as f:
        hf_token = f.read().strip()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

image_path = "images/1.5umPS_0.4umPCTE+30sAu_X100_brightfield_1.jpg"
prompt = "particle"
# ---- FIX: force HF auth ----
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, text=prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print("type(outputs):", type(outputs))

if hasattr(outputs, "keys"):
    print("outputs.keys():", outputs.keys())

for attr in ["pred_masks", "iou_scores", "mask_logits", "vision_outputs"]:
    if hasattr(outputs, attr):
        val = getattr(outputs, attr)
        if hasattr(val, "shape"):
            print(f"{attr}.shape:", val.shape)
        else:
            print(f"{attr}:", type(val))

print(outputs)