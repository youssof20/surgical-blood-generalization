"""
Phase 1 — Data pipeline for HemoSet (train domain) and CholecSeg8k (OOD test).

Paths are relative to the project root (parent of ``src/``).
"""

from __future__ import annotations

import glob
import json
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HEMOSET = PROJECT_ROOT / "data" / "hemoset"
DATA_CHOLEC = PROJECT_ROOT / "data" / "cholecseg8k"
OUT_RESULTS = PROJECT_ROOT / "outputs" / "results"

IMG_SIZE = 256
RNG_SEED = 42

# ImageNet normalization (for consistency with later training; stats below use raw RGB)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)

# CholecSeg8k: blood class in color masks — RGB (231, 70, 156); BGR for OpenCV inRange.
CHOLEC_BLOOD_RGB = np.array([231, 70, 156], dtype=np.uint8)
CHOLEC_BLOOD_BGR = np.array([156, 70, 231], dtype=np.uint8)


@dataclass(frozen=True)
class HemoSample:
    pig: str
    stem: str  # e.g. "001020" — matches imgs/{stem}.png and labels/{stem}_mask.png


def _ensure_dirs() -> None:
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)


def discover_hemoset_samples() -> list[HemoSample]:
    """Pair images and masks from ``pig*/imgs.zip`` + ``labels.zip``."""
    samples: list[HemoSample] = []
    for pig_dir in sorted(DATA_HEMOSET.glob("pig*")):
        if not pig_dir.is_dir():
            continue
        pig = pig_dir.name
        iz = pig_dir / "imgs.zip"
        lz = pig_dir / "labels.zip"
        if not iz.is_file() or not lz.is_file():
            continue
        with zipfile.ZipFile(iz) as zimg, zipfile.ZipFile(lz) as zlab:
            img_names = {Path(n).name for n in zimg.namelist() if n.startswith("imgs/") and n.endswith(".png")}
            lab_names = {Path(n).name for n in zlab.namelist() if n.startswith("labels/") and n.endswith("_mask.png")}
        stems = sorted(
            Path(n).stem.replace("_mask", "")
            for n in lab_names
            if f"{Path(n).stem.replace('_mask', '')}.png" in img_names
        )
        for stem in stems:
            img_entry = f"imgs/{stem}.png"
            lab_entry = f"labels/{stem}_mask.png"
            with zipfile.ZipFile(iz) as zimg, zipfile.ZipFile(lz) as zlab:
                if img_entry not in zimg.namelist() or lab_entry not in zlab.namelist():
                    continue
            samples.append(HemoSample(pig=pig, stem=stem))
    if not samples:
        raise FileNotFoundError(
            f"No HemoSet samples found under {DATA_HEMOSET}. "
            "Expected pig*/imgs.zip and pig*/labels.zip with paired imgs/*.png and labels/*_mask.png."
        )
    return samples


def _read_png_from_zip(zip_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode {member} in {zip_path}")
    return img


def hemoset_paths(sample: HemoSample) -> tuple[Path, str, str]:
    base = DATA_HEMOSET / sample.pig
    return base, f"imgs/{sample.stem}.png", f"labels/{sample.stem}_mask.png"


def load_hemoset_image_mask(sample: HemoSample) -> tuple[np.ndarray, np.ndarray]:
    """RGB image (H,W,3) uint8 and binary blood mask (H,W) float32 in {0,1}."""
    zpath, imem, mmem = hemoset_paths(sample)
    img_bgr = _read_png_from_zip(zpath / "imgs.zip", imem)
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask = _read_png_from_zip(zpath / "labels.zip", mmem)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # HemoSet exports vary; treat any positive label as blood.
    blood = (mask > 0).astype(np.float32)
    return img_rgb, blood


def discover_cholecseg8k_frames() -> list[Path]:
    """All ``frame_*_endo.png`` (raw RGB), excluding mask-only files."""
    pattern = str(DATA_CHOLEC / "**" / "frame_*_endo.png")
    frames = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    if not frames:
        raise FileNotFoundError(
            f"No CholecSeg8k frames found under {DATA_CHOLEC}. "
            "Expected **/frame_*_endo.png (Kaggle layout)."
        )
    return frames


def cholec_color_mask_path(frame_endo: Path) -> Path:
    """frame_123_endo.png -> frame_123_endo_color_mask.png"""
    stem = frame_endo.stem  # frame_123_endo
    return frame_endo.with_name(f"{stem}_color_mask.png")


def load_cholec_image_mask(frame_endo: Path) -> tuple[np.ndarray, np.ndarray]:
    """RGB image and binary blood mask (float32 {0,1}) from color mask."""
    cpath = cholec_color_mask_path(frame_endo)
    if not cpath.is_file():
        raise FileNotFoundError(f"Missing color mask: {cpath}")

    img_bgr = cv2.imread(str(frame_endo), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {frame_endo}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    cm_bgr = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
    if cm_bgr is None:
        raise ValueError(f"Failed to read color mask: {cpath}")
    # cv2.inRange is much faster than np.all on full-res masks (avoids huge bool reductions).
    lo = CHOLEC_BLOOD_BGR
    hi = CHOLEC_BLOOD_BGR
    blood_u8 = cv2.inRange(cm_bgr, lo, hi)
    blood = (blood_u8 > 0).astype(np.float32)
    return img_rgb, blood


def resize_both(
    img_rgb: np.ndarray, mask: np.ndarray, size: int = IMG_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    img_r = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, m


def normalize_imagenet(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """float32 CHW tensor, ImageNet-normalized."""
    x = img_rgb_uint8.astype(np.float64) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1).astype(np.float32)


def image_stats_raw(img_rgb_256: np.ndarray) -> dict[str, float]:
    """Per-image RGB means, grayscale brightness, HSV saturation mean."""
    p = img_rgb_256.astype(np.float64) / 255.0
    r, g, b = p[..., 0].mean(), p[..., 1].mean(), p[..., 2].mean()
    gray = cv2.cvtColor(img_rgb_256, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    mean_brightness = float(gray.mean())
    hsv = cv2.cvtColor(img_rgb_256, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float64) / 255.0
    mean_saturation = float(sat.mean())
    return {
        "mean_R": float(r),
        "mean_G": float(g),
        "mean_B": float(b),
        "std_R": float(p[..., 0].std()),
        "std_G": float(p[..., 1].std()),
        "std_B": float(p[..., 2].std()),
        "mean_brightness": mean_brightness,
        "mean_saturation": mean_saturation,
    }


def aggregate_domain_stats(per_image: list[dict[str, float]]) -> dict[str, float]:
    """Mean of per-image summary stats (typical domain summary)."""
    keys = per_image[0].keys()
    out: dict[str, float] = {}
    n = len(per_image)
    for k in keys:
        out[k] = float(sum(d[k] for d in per_image) / n)
    return out


def run_phase1() -> dict[str, Any]:
    _ensure_dirs()
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    hemo_samples = discover_hemoset_samples()
    ids = [f"{s.pig}/{s.stem}" for s in hemo_samples]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(0.8 * n)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:])

    split_payload = {
        "seed": RNG_SEED,
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
    }
    (OUT_RESULTS / "hemoset_splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    # --- HemoSet: coverage + domain stats on train split
    hemo_train_stats: list[dict[str, float]] = []
    hemo_blood_pcts: list[float] = []
    id_to_sample = {f"{s.pig}/{s.stem}": s for s in hemo_samples}

    for sid in sorted(train_ids):
        s = id_to_sample[sid]
        img, m = load_hemoset_image_mask(s)
        img, m = resize_both(img, m)
        hemo_blood_pcts.append(float(m.mean() * 100.0))
        hemo_train_stats.append(image_stats_raw(img))

    hemo_val_blood: list[float] = []
    for sid in sorted(val_ids):
        s = id_to_sample[sid]
        img, m = load_hemoset_image_mask(s)
        _, m = resize_both(img, m)
        hemo_val_blood.append(float(m.mean() * 100.0))

    hemo_train_domain = aggregate_domain_stats(hemo_train_stats)

    # --- CholecSeg8k (full test set)
    cholec_frames = discover_cholecseg8k_frames()
    print(f"  Processing {len(cholec_frames)} CholecSeg8k frames (this may take a few minutes)...", flush=True)
    cholec_per_image: list[dict[str, float]] = []
    frames_with_blood = 0
    blood_pcts: list[float] = []

    n_chol = len(cholec_frames)
    for i, fp in enumerate(cholec_frames):
        if i % 500 == 0 or i == n_chol - 1:
            print(f"  CholecSeg8k: {i + 1}/{n_chol} frames...", flush=True)
        img, m = load_cholec_image_mask(fp)
        img, m = resize_both(img, m)
        cov = float(m.mean() * 100.0)
        blood_pcts.append(cov)
        if m.max() > 0:
            frames_with_blood += 1
        cholec_per_image.append(image_stats_raw(img))

    cholec_domain = aggregate_domain_stats(cholec_per_image)

    blood_stats = {
        "total_frames": len(cholec_frames),
        "frames_with_blood": frames_with_blood,
        "frames_without_blood": len(cholec_frames) - frames_with_blood,
        "blood_pixel_pct_mean": float(np.mean(blood_pcts)) if blood_pcts else 0.0,
        "blood_class_rgb": CHOLEC_BLOOD_RGB.tolist(),
        "note": "Blood pixels match RGB (231,70,156) in color masks (class Blood, watershed ID 13 in samples).",
    }
    (OUT_RESULTS / "cholecseg8k_blood_stats.json").write_text(
        json.dumps(blood_stats, indent=2), encoding="utf-8"
    )

    domain_stats = {
        "image_size": IMG_SIZE,
        "imagenet_mean": IMAGENET_MEAN.tolist(),
        "imagenet_std": IMAGENET_STD.tolist(),
        "hemoset_train": {
            **hemo_train_domain,
            "n_images": len(train_ids),
            "mean_blood_coverage_pct": float(np.mean(hemo_blood_pcts)) if hemo_blood_pcts else 0.0,
        },
        "cholecseg8k_test": {
            **cholec_domain,
            "n_images": len(cholec_frames),
            "frames_with_blood": frames_with_blood,
            "frames_without_blood": len(cholec_frames) - frames_with_blood,
            "mean_blood_coverage_pct": float(np.mean(blood_pcts)) if blood_pcts else 0.0,
        },
    }
    (OUT_RESULTS / "domain_stats.json").write_text(json.dumps(domain_stats, indent=2), encoding="utf-8")

    # --- Console report
    print("=== HemoSet ===")
    print(f"  Total paired samples: {n}")
    print(f"  Train: {len(train_ids)}  |  Val: {len(val_ids)}")
    print(f"  Mean blood coverage % (train, per image): {np.mean(hemo_blood_pcts):.4f}")
    print(f"  Mean blood coverage % (val, per image):   {np.mean(hemo_val_blood):.4f}")

    print("\n=== CholecSeg8k (OOD test, not used for training) ===")
    print(f"  N test frames: {len(cholec_frames)}")
    print(f"  Frames with blood: {frames_with_blood}  |  without blood: {len(cholec_frames) - frames_with_blood}")
    print(f"  Mean blood coverage % (per frame): {np.mean(blood_pcts):.4f}")

    print("\n=== Domain gap (mean of per-image stats @ 256x256, raw RGB) ===")
    print("  HemoSet train — R,G,B brightness sat: "
          f"{hemo_train_domain['mean_R']:.4f}, {hemo_train_domain['mean_G']:.4f}, "
          f"{hemo_train_domain['mean_B']:.4f}, {hemo_train_domain['mean_brightness']:.4f}, "
          f"{hemo_train_domain['mean_saturation']:.4f}")
    print("  CholecSeg8k   — R,G,B brightness sat: "
          f"{cholec_domain['mean_R']:.4f}, {cholec_domain['mean_G']:.4f}, "
          f"{cholec_domain['mean_B']:.4f}, {cholec_domain['mean_brightness']:.4f}, "
          f"{cholec_domain['mean_saturation']:.4f}")

    return {
        "split": split_payload,
        "blood_stats": blood_stats,
        "domain_stats": domain_stats,
    }


def main() -> None:
    run_phase1()


if __name__ == "__main__":
    main()
