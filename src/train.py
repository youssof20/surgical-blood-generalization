"""
Phase 2 — Train blood segmentation on HemoSet train split; evaluate on val + CholecSeg8k OOD.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

from src.data_pipeline import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OUT_RESULTS,
    PROJECT_ROOT,
    HemoSample,
    discover_cholecseg8k_frames,
    load_cholec_image_mask,
    load_hemoset_image_mask,
    normalize_imagenet,
    resize_both,
)
from src.models import build_unet

# -----------------------------------------------------------------------------
# Paths & hyperparameters
# -----------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SPLITS_PATH = OUT_RESULTS / "hemoset_splits.json"

BEST_CKPT = MODELS_DIR / "unet_resnet34_best.pt"
RESULTS_JSON = OUT_RESULTS / "training_results.json"
CURVES_PNG = FIGURES_DIR / "training_curves.png"

RNG_SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
T_MAX = 30
ETA_MIN = 1e-6
EARLY_PATIENCE = 10
NUM_WORKERS = 0


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_id_to_sample(sid: str) -> HemoSample:
    pig, stem = sid.split("/", 1)
    return HemoSample(pig=pig, stem=stem)


def _train_augment() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"mask": "mask"},
    )


def _val_augment() -> A.Compose:
    return A.Compose(
        [A.Resize(IMG_SIZE, IMG_SIZE)],
        additional_targets={"mask": "mask"},
    )


class HemosetDataset(Dataset):
    """HemoSet samples from ``hemoset_splits.json`` (train or val)."""

    def __init__(self, ids: list[str], train: bool) -> None:
        self.samples = [split_id_to_sample(s) for s in ids]
        self.train = train
        self.aug = _train_augment() if train else _val_augment()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img_rgb, mask = load_hemoset_image_mask(s)
        mask_u8 = (mask * 255).astype(np.uint8)
        out = self.aug(image=img_rgb, mask=mask_u8)
        img_rgb = out["image"]
        mask = out["mask"].astype(np.float32) / 255.0
        if mask.ndim == 2:
            pass
        else:
            mask = mask[..., 0]
        x = normalize_imagenet(img_rgb)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return xt, yt


class CholecEvalDataset(Dataset):
    """Full CholecSeg8k — OOD evaluation only."""

    def __init__(self, frames: list[Path]) -> None:
        self.frames = frames
        self.aug = _val_augment()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fp = self.frames[idx]
        img_rgb, mask = load_cholec_image_mask(fp)
        mask_u8 = (mask * 255).astype(np.uint8)
        out = self.aug(image=img_rgb, mask=mask_u8)
        img_rgb = out["image"]
        mask = out["mask"].astype(np.float32) / 255.0
        if mask.ndim == 2:
            pass
        else:
            mask = mask[..., 0]
        x = normalize_imagenet(img_rgb)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return xt, yt


@torch.no_grad()
def binary_metrics(
    pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5
) -> tuple[float, float, float, float]:
    """Batch-wise micro-averaged Dice, IoU, precision, recall."""
    p = (pred > thr).float()
    t = (target > thr).float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return float(dice.item()), float(iou.item()), float(prec.item()), float(rec.item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    criterion_dice: nn.Module,
    criterion_bce: nn.Module,
    train: bool,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    sum_dice = sum_iou = sum_prec = sum_rec = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = 0.5 * criterion_dice(logits, y) + 0.5 * criterion_bce(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        d, i, pr, re = binary_metrics(logits, y)
        sum_dice += d
        sum_iou += i
        sum_prec += pr
        sum_rec += re
        n_batches += 1

    return {
        "loss": float(np.mean(losses)),
        "dice": sum_dice / max(n_batches, 1),
        "iou": sum_iou / max(n_batches, 1),
        "precision": sum_prec / max(n_batches, 1),
        "recall": sum_rec / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0.0
    total_loss = 0.0
    n_batches = 0
    criterion_dice = smp.losses.DiceLoss(mode="binary", from_logits=False)
    criterion_bce = nn.BCELoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = 0.5 * criterion_dice(pred, y) + 0.5 * criterion_bce(pred, y)
        total_loss += float(loss.item())
        p = (pred > 0.5).float()
        t = (y > 0.5).float()
        tp += (p * t).sum().item()
        fp += (p * (1 - t)).sum().item()
        fn += ((1 - p) * t).sum().item()
        n_batches += 1

    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
    }


def plot_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_dice"], label="train")
    ax.plot(epochs, history["val_dice"], label="val")
    ax.set_title("Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["train_iou"], label="train")
    ax.plot(epochs, history["val_iou"], label="val")
    ax.set_title("IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="tab:purple")
    ax.set_title("Learning rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_training() -> dict[str, Any]:
    _ensure_dirs()
    set_seed(RNG_SEED)

    if not SPLITS_PATH.is_file():
        raise FileNotFoundError(f"Missing {SPLITS_PATH}. Run Phase 1 (src.data_pipeline) first.")

    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids: list[str] = splits["train"]
    val_ids: list[str] = splits["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()

    train_ds = HemosetDataset(train_ids, train=True)
    val_ds = HemosetDataset(val_ids, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    cholec_frames = discover_cholecseg8k_frames()
    cholec_ds = CholecEvalDataset(cholec_frames)
    cholec_loader = DataLoader(
        cholec_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    model = build_unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )
    criterion_dice = smp.losses.DiceLoss(mode="binary", from_logits=False)
    criterion_bce = nn.BCELoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "lr": [],
    }

    best_val_dice = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion_dice,
            criterion_bce,
            train=True,
        )
        va = run_epoch(
            model,
            val_loader,
            device,
            None,
            criterion_dice,
            criterion_bce,
            train=False,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_dice"].append(tr["dice"])
        history["val_dice"].append(va["dice"])
        history["train_iou"].append(tr["iou"])
        history["val_iou"].append(va["iou"])
        history["lr"].append(lr)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS}  train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
            f"val_Dice={va['dice']:.4f}  val_IoU={va['iou']:.4f}  lr={lr:.2e}",
            flush=True,
        )

        if va["dice"] > best_val_dice:
            best_val_dice = va["dice"]
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": va["dice"],
                "val_iou": va["iou"],
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val Dice improvement for {EARLY_PATIENCE} epochs).", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")

    torch.save(best_state, BEST_CKPT)
    model.load_state_dict(best_state["model_state_dict"])

    hemoset_val = evaluate_loader(model, val_loader, device)
    cholec = evaluate_loader(model, cholec_loader, device)

    hd = hemoset_val["dice"]
    hi = hemoset_val["iou"]
    cd = cholec["dice"]
    ci = cholec["iou"]
    dice_drop_pct = float((hd - cd) / max(hd, 1e-8) * 100.0)
    iou_drop_pct = float((hi - ci) / max(hi, 1e-8) * 100.0)

    results: dict[str, Any] = {
        "best_epoch": best_epoch,
        "hemoset_val_dice": hd,
        "hemoset_val_iou": hi,
        "hemoset_val_precision": hemoset_val["precision"],
        "hemoset_val_recall": hemoset_val["recall"],
        "cholecseg8k_dice": cd,
        "cholecseg8k_iou": ci,
        "cholecseg8k_precision": cholec["precision"],
        "cholecseg8k_recall": cholec["recall"],
        "dice_drop_pct": dice_drop_pct,
        "iou_drop_pct": iou_drop_pct,
        "epochs_trained": len(history["train_loss"]),
        "early_stopped": epochs_no_improve >= EARLY_PATIENCE,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_curves(history, CURVES_PNG)

    print("\nIn-domain (HemoSet val) Dice: {:.4f}".format(hd), flush=True)
    print("Out-of-domain (CholecSeg8k) Dice: {:.4f}".format(cd), flush=True)
    print("Performance drop (Dice): {:.2f}%".format(dice_drop_pct), flush=True)

    return {"results": results, "history": history}


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
