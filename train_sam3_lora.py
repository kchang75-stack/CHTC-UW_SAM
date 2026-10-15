#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import Sam3Model, Sam3Processor


class BinaryMaskDataset(Dataset):
    def __init__(self, root: str, image_size: int = 320):
        self.root = Path(root)
        self.image_size = image_size
        self.images = sorted((self.root / "images").glob("*.png"))
        if not self.images:
            raise RuntimeError(f"No images found in {self.root / 'images'}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image_path = self.images[idx]
        mask_path = self.root / "masks" / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image: {image_path} -> {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        mask_np = (np.array(mask) > 0).astype(np.float32)

        return {
            "image": image,
            "mask": torch.from_numpy(mask_np),
            "name": image_path.stem,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "images": [b["image"] for b in batch],
        "masks": torch.stack([b["mask"] for b in batch], dim=0),
        "names": [b["name"] for b in batch],
    }


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def get_hf_token() -> str | None:
    token_path = Path("hf_token.txt")
    if token_path.exists():
        tok = token_path.read_text().strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGINGFACE_HUB_TOKEN"] = tok
            return tok
    return None


def choose_target_modules(model: torch.nn.Module) -> List[str]:
    # Most conservative memory choice
    return ["q_proj"]


def compute_loss(outputs, targets: torch.Tensor) -> torch.Tensor:
    """
    Version 1 LoRA target:
    use semantic_seg to learn a union mask per image.
    """
    if not hasattr(outputs, "semantic_seg"):
        raise RuntimeError("SAM3 outputs do not contain semantic_seg")

    logits = outputs.semantic_seg  # expected [B, 1, H, W]
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise RuntimeError(f"Unexpected semantic_seg shape: {tuple(logits.shape)}")

    logits = F.interpolate(
        logits,
        size=targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


@torch.no_grad()
def evaluate(model, processor, loader, prompt: str, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    amp_device = "cuda" if device == "cuda" else "cpu"

    for batch in loader:
        inputs = processor(images=batch["images"], text=prompt, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        targets = batch["masks"].to(device, non_blocking=True)

        with autocast(device_type=amp_device, enabled=(device == "cuda")):
            outputs = model(**inputs)
            loss = compute_loss(outputs, targets)

        total += float(loss.item())
        count += 1

        del inputs, targets, outputs, loss

    model.train()
    return total / max(count, 1)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_peft_checkpoint(model, processor, ckpt_dir: Path, extra_metrics: Dict | None = None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to: {ckpt_dir.resolve()}", flush=True)

    model.save_pretrained(ckpt_dir, safe_serialization=True)
    processor.save_pretrained(ckpt_dir)

    if extra_metrics is not None:
        write_json(ckpt_dir / "metrics.json", extra_metrics)

    print("Checkpoint contents after save:", flush=True)
    for p in sorted(ckpt_dir.iterdir()):
        print(f"  - {p.name}", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="lora_dataset/train")
    parser.add_argument("--val-root", default="lora_dataset/val")
    parser.add_argument("--output-dir", default="sam3_lora_out")
    parser.add_argument("--prompt", default="particle")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("output_dir:", output_dir.resolve(), flush=True)

    hf_token = get_hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device = "cuda" if device == "cuda" else "cpu"

    print("device:", device, flush=True)
    print("hf token present:", bool(hf_token), flush=True)

    # Slight perf improvement on supported NVIDIA GPUs; doesn't reduce memory much.
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds = BinaryMaskDataset(args.train_root, image_size=args.image_size)
    val_ds = BinaryMaskDataset(args.val_root, image_size=args.image_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    print(f"train images: {len(train_ds)}", flush=True)
    print(f"val images: {len(val_ds)}", flush=True)

    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    base_model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)

    # Gradient checkpointing to lower activation memory
    # Sam3Model does not support gradient checkpointing
    print("gradient checkpointing: not supported for Sam3Model", flush=True)

    if hasattr(base_model, "config") and hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
        print("config.use_cache: False", flush=True)

    target_modules = choose_target_modules(base_model)
    print("LoRA target modules:", target_modules, flush=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(base_model, lora_cfg).to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(device=amp_device, enabled=(device == "cuda"))

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dl, start=1):
            inputs = processor(images=batch["images"], text=args.prompt, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            targets = batch["masks"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=amp_device, enabled=(device == "cuda")):
                outputs = model(**inputs)
                loss = compute_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())

            if step % 2 == 0 or step == len(train_dl):
                print(
                    f"epoch={epoch + 1} step={step}/{len(train_dl)} loss={loss.item():.4f}",
                    flush=True,
                )

            del inputs, targets, outputs, loss

        train_loss = epoch_loss / max(len(train_dl), 1)
        val_loss = evaluate(model, processor, val_dl, args.prompt, device)

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "prompt": args.prompt,
            "image_size": args.image_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
        }

        if ((epoch + 1) % args.save_every) == 0:
            ckpt_dir = output_dir / f"epoch_{epoch + 1}"
            save_peft_checkpoint(model, processor, ckpt_dir, extra_metrics=metrics)

        if val_loss < best_val:
            best_val = val_loss
            best_dir = output_dir / "best"
            save_peft_checkpoint(model, processor, best_dir, extra_metrics=metrics)

            with open(best_dir / "metrics.txt", "w") as f:
                f.write(
                    f"epoch={epoch + 1}\n"
                    f"train_loss={train_loss:.6f}\n"
                    f"val_loss={val_loss:.6f}\n"
                )

    print("Final output directory tree:", flush=True)
    for p in sorted(output_dir.rglob("*")):
        print(f"  {p}", flush=True)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()