"""
Phase 3 — Per-frame OOD analysis, failure modes, Spearman correlations,
and Gaussian Fréchet-style distance between HemoSet train and CholecSeg8k.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset

import albumentations as A

from src.data_pipeline import (
    DATA_CHOLEC,
    IMG_SIZE,
    OUT_RESULTS,
    discover_cholecseg8k_frames,
    image_stats_raw,
    load_cholec_image_mask,
    load_hemoset_image_mask,
    normalize_imagenet,
    resize_both,
)
from src.models import build_unet
from src.train import BEST_CKPT, NUM_WORKERS, SPLITS_PATH, split_id_to_sample

DOMAIN_STATS_PATH = OUT_RESULTS / "domain_stats.json"
RESULTS_JSON = OUT_RESULTS / "training_results.json"
PER_FRAME_CSV = OUT_RESULTS / "per_frame_analysis.csv"
CORRELATION_CSV = OUT_RESULTS / "correlation_analysis.csv"

BATCH_SIZE = 16
PRED_THRESHOLD = 0.5


def _ensure_dirs() -> None:
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)


def _val_resize() -> A.Compose:
    return A.Compose(
        [A.Resize(IMG_SIZE, IMG_SIZE)],
        additional_targets={"mask": "mask"},
    )


class CholecAnalysisDataset(Dataset):
    """CholecSeg8k: tensor + mask + frame id + raw stats (256×256 RGB)."""

    def __init__(self, frames: list[Path], cholec_root: Path) -> None:
        self.frames = frames
        self.cholec_root = cholec_root
        self.aug = _val_resize()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple:
        fp = self.frames[idx]
        img_rgb, mask = load_cholec_image_mask(fp)
        mask_u8 = (mask * 255).astype(np.uint8)
        out = self.aug(image=img_rgb, mask=mask_u8)
        img_rgb = out["image"]
        mask = out["mask"].astype(np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[..., 0]
        stats_dict = image_stats_raw(img_rgb)
        x = normalize_imagenet(img_rgb)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        try:
            rel = fp.relative_to(self.cholec_root)
            frame_id = str(rel).replace("\\", "/")
        except ValueError:
            frame_id = str(fp.name)
        return xt, yt, frame_id, stats_dict


def collate_analysis(batch: list) -> tuple:
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    ids = [b[2] for b in batch]
    stats_list = [b[3] for b in batch]
    return xs, ys, ids, stats_list


def dice_iou_per_image(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> tuple[float, float]:
    """pred, gt: (1,H,W) on device."""
    p = (pred > PRED_THRESHOLD).float()
    t = (gt > PRED_THRESHOLD).float()
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    fn = ((1.0 - p) * t).sum()
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return float(dice.item()), float(iou.item())


def failure_mode(dice: float, has_blood: bool, pred: torch.Tensor) -> str:
    """Single label per frame (see project spec)."""
    has_pred_blood = bool((pred > PRED_THRESHOLD).any().item())

    if not has_blood:
        return "FALSE_POSITIVE" if has_pred_blood else "CORRECT"

    if dice >= 0.5:
        return "CORRECT"
    if dice > 0.0:
        return "PARTIAL"
    return "FALSE_NEGATIVE"


def collect_hemoset_train_features() -> np.ndarray:
    """N × 5 matrix: mean_R, mean_G, mean_B, mean_brightness, mean_saturation @ 256²."""
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids: list[str] = splits["train"]
    rows: list[list[float]] = []
    for sid in train_ids:
        s = split_id_to_sample(sid)
        img_rgb, mask = load_hemoset_image_mask(s)
        img_rgb, _ = resize_both(img_rgb, mask)
        st = image_stats_raw(img_rgb)
        rows.append(
            [
                st["mean_R"],
                st["mean_G"],
                st["mean_B"],
                st["mean_brightness"],
                st["mean_saturation"],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def frechet_gaussian(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between N(μ₁,Σ₁) and N(μ₂,Σ₂) fitted to samples (FID-style)."""
    n1, d = X.shape
    n2, _ = Y.shape
    mu1 = X.mean(axis=0)
    mu2 = Y.mean(axis=0)
    sigma1 = np.cov(X, rowvar=False)
    sigma2 = np.cov(Y, rowvar=False)
    if sigma1.ndim == 0:
        sigma1 = np.array([[float(sigma1)]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[float(sigma2)]])
    sigma1 = sigma1 + np.eye(d) * eps
    sigma2 = sigma2 + np.eye(d) * eps

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


@torch.no_grad()
def run_phase3() -> dict[str, Any]:
    _ensure_dirs()

    if not BEST_CKPT.is_file():
        raise FileNotFoundError(f"Missing checkpoint {BEST_CKPT}. Train the model first (Phase 2).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model = build_unet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frames = discover_cholecseg8k_frames()
    ds = CholecAnalysisDataset(frames, DATA_CHOLEC)
    pin = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        collate_fn=collate_analysis,
    )

    rows_out: list[dict[str, Any]] = []
    n_frames = len(frames)
    bi = 0

    for xs, ys, frame_ids, stats_list in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        preds = model(xs)
        bsz = xs.shape[0]
        for j in range(bsz):
            pred = preds[j : j + 1]
            gt = ys[j : j + 1]
            d, iou = dice_iou_per_image(pred, gt)
            st = stats_list[j]
            gt_cpu = gt.cpu()
            has_blood = bool((gt_cpu > PRED_THRESHOLD).any().item())
            blood_cov = float(gt_cpu.mean().item() * 100.0)
            mode = failure_mode(d, has_blood, pred)

            rows_out.append(
                {
                    "frame_id": frame_ids[j],
                    "dice": d,
                    "iou": iou,
                    "mean_brightness": st["mean_brightness"],
                    "mean_saturation": st["mean_saturation"],
                    "mean_R": st["mean_R"],
                    "mean_G": st["mean_G"],
                    "mean_B": st["mean_B"],
                    "blood_coverage_pct": blood_cov,
                    "has_blood": has_blood,
                    "failure_mode": mode,
                }
            )
        bi += bsz
        if bi % 500 == 0 or bi >= n_frames:
            print(f"  CholecSeg8k inference: {bi}/{n_frames} frames...", flush=True)

    fieldnames = [
        "frame_id",
        "dice",
        "iou",
        "mean_brightness",
        "mean_saturation",
        "mean_R",
        "mean_G",
        "mean_B",
        "blood_coverage_pct",
        "has_blood",
        "failure_mode",
    ]
    with PER_FRAME_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            r = dict(row)
            r["has_blood"] = str(r["has_blood"]).lower()
            w.writerow(r)

    # Failure mode counts
    from collections import Counter

    counts = Counter(r["failure_mode"] for r in rows_out)
    failure_payload = {k: int(v) for k, v in counts.items()}

    if RESULTS_JSON.is_file():
        tr = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    else:
        tr = {}
    tr["failure_modes"] = failure_payload
    RESULTS_JSON.write_text(json.dumps(tr, indent=2), encoding="utf-8")

    # Spearman (GT blood frames only)
    feats = (
        "mean_brightness",
        "mean_saturation",
        "mean_R",
        "mean_G",
        "mean_B",
        "blood_coverage_pct",
    )
    blood_rows = [r for r in rows_out if r["has_blood"]]
    dice_vec = np.array([r["dice"] for r in blood_rows], dtype=np.float64)
    corr_rows: list[dict[str, Any]] = []

    for feat in feats:
        xvec = np.array([r[feat] for r in blood_rows], dtype=np.float64)
        if len(xvec) < 3 or np.std(xvec) < 1e-12 or np.std(dice_vec) < 1e-12:
            corr_rows.append({"feature": feat, "spearman_r": float("nan"), "p_value": float("nan")})
        else:
            r_s, p_v = stats.spearmanr(xvec, dice_vec)
            corr_rows.append(
                {
                    "feature": feat,
                    "spearman_r": float(r_s) if not np.isnan(r_s) else float("nan"),
                    "p_value": float(p_v) if not np.isnan(p_v) else float("nan"),
                }
            )

    with CORRELATION_CSV.open("w", newline="", encoding="utf-8") as f:
        cw = csv.DictWriter(f, fieldnames=["feature", "spearman_r", "p_value"])
        cw.writeheader()
        cw.writerows(corr_rows)

    # Fréchet distance: HemoSet train vs CholecSeg8k (per-image 5-D stats)
    print("  Computing HemoSet train feature matrix for Fréchet distance...", flush=True)
    X_hemo = collect_hemoset_train_features()
    X_chol = np.array(
        [[r["mean_R"], r["mean_G"], r["mean_B"], r["mean_brightness"], r["mean_saturation"]] for r in rows_out],
        dtype=np.float64,
    )
    fd = frechet_gaussian(X_hemo, X_chol)

    if DOMAIN_STATS_PATH.is_file():
        dom = json.loads(DOMAIN_STATS_PATH.read_text(encoding="utf-8"))
    else:
        dom = {}
    dom["frechet_distance_gaussian"] = {
        "value": fd,
        "description": "Fréchet distance between N(μ,Σ) fitted to per-image "
        "[mean_R, mean_G, mean_B, mean_brightness, mean_saturation] (256×256, same as Phase 1).",
        "n_hemoset_train": int(X_hemo.shape[0]),
        "n_cholecseg8k": int(X_chol.shape[0]),
        "feature_order": ["mean_R", "mean_G", "mean_B", "mean_brightness", "mean_saturation"],
    }
    DOMAIN_STATS_PATH.write_text(json.dumps(dom, indent=2), encoding="utf-8")

    print("\nFailure mode counts:", json.dumps(failure_payload, indent=2))
    print(f"Gaussian Fréchet distance (5-D stats): {fd:.6f}")
    print(f"Wrote {PER_FRAME_CSV}")
    print(f"Wrote {CORRELATION_CSV}")
    print(f"Updated {RESULTS_JSON} (failure_modes)")
    print(f"Updated {DOMAIN_STATS_PATH} (frechet_distance_gaussian)")
    print("PHASE 3 COMPLETE", flush=True)

    return {
        "failure_modes": failure_payload,
        "frechet_distance": fd,
        "n_per_frame_rows": len(rows_out),
    }


def main() -> None:
    run_phase3()


if __name__ == "__main__":
    main()
