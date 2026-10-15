#!/usr/bin/env python3
from __future__ import annotations

from transformers import Sam3Model

model = Sam3Model.from_pretrained("facebook/sam3")

for name, module in model.named_modules():
    cls = module.__class__.__name__.lower()
    if "attention" in name.lower() or "attn" in name.lower() or cls in {"linear"}:
        print(name, "->", module.__class__.__name__)