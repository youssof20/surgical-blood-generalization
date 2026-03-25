"""
Phase 4 — Publication figures (5 PNGs) for domain gap, metrics, failure modes,
correlations, and qualitative OOD examples.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_pipeline import (
    DATA_CHOLEC,
    IMG_SIZE,
    OUT_RESULTS,
    PROJECT_ROOT,
    discover_cholecseg8k_frames,
    image_stats_raw,
    load_cholec_image_mask,
    load_hemoset_image_mask,
    normalize_imagenet,
    resize_both,
)
from src.models import build_unet
from src.train import BEST_CKPT, SPLITS_PATH, split_id_to_sample

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_JSON = OUT_RESULTS / "training_results.json"
CORRELATION_CSV = OUT_RESULTS / "correlation_analysis.csv"
PER_FRAME_CSV = OUT_RESULTS / "per_frame_analysis.csv"

RNG_SEED = 42
DPI = 150


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _mean_rgb_caption(images_rgb: list[np.ndarray]) -> str:
    """Average mean R,G,B in [0,1] across images (each image full mean)."""
    rs, gs, bs = [], [], []
    for im in images_rgb:
        p = im.astype(np.float64) / 255.0
        rs.append(p[..., 0].mean())
        gs.append(p[..., 1].mean())
        bs.append(p[..., 2].mean())
    return f"mean RGB = ({np.mean(rs):.3f}, {np.mean(gs):.3f}, {np.mean(bs):.3f})"


def figure_domain_gap() -> None:
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids = splits["train"]
    pick_train = random.sample(train_ids, min(4, len(train_ids)))
    hemo_rgb: list[np.ndarray] = []
    for sid in pick_train:
        s = split_id_to_sample(sid)
        img, m = load_hemoset_image_mask(s)
        img, _ = resize_both(img, m)
        hemo_rgb.append(img)

    cholec_all = discover_cholecseg8k_frames()
    pick_ch = random.sample(cholec_all, min(4, len(cholec_all)))
    cho_rgb: list[np.ndarray] = []
    for fp in pick_ch:
        img, m = load_cholec_image_mask(fp)
        img, _ = resize_both(img, m)
        cho_rgb.append(img)

    _set_style()
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(
        "Training Domain (Robotic Surgery) vs Test Domain (Laparoscopic Surgery)",
        fontsize=12,
        y=0.98,
    )
    gs_l = fig.add_gridspec(2, 2, left=0.04, right=0.46, top=0.82, bottom=0.14, wspace=0.08, hspace=0.12)
    for i in range(4):
        ax = fig.add_subplot(gs_l[i // 2, i % 2])
        ax.imshow(hemo_rgb[i])
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("0.7")
    cap_h = _mean_rgb_caption(hemo_rgb)
    fig.text(0.25, 0.04, f"HemoSet (train)\n{cap_h}", ha="center", fontsize=9)

    gs_r = fig.add_gridspec(2, 2, left=0.54, right=0.96, top=0.82, bottom=0.14, wspace=0.08, hspace=0.12)
    for i in range(4):
        ax = fig.add_subplot(gs_r[i // 2, i % 2])
        ax.imshow(cho_rgb[i])
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("0.7")
    cap_c = _mean_rgb_caption(cho_rgb)
    fig.text(0.75, 0.04, f"CholecSeg8k (OOD test)\n{cap_c}", ha="center", fontsize=9)

    fig.savefig(FIGURES_DIR / "domain_gap_visual.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure_performance() -> None:
    r = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    metrics = [
        ("Dice", "hemoset_val_dice", "cholecseg8k_dice"),
        ("IoU", "hemoset_val_iou", "cholecseg8k_iou"),
        ("Precision", "hemoset_val_precision", "cholecseg8k_precision"),
        ("Recall", "hemoset_val_recall", "cholecseg8k_recall"),
    ]
    names = [m[0] for m in metrics]
    in_vals = [float(r[m[1]]) for m in metrics]
    out_vals = [float(r[m[2]]) for m in metrics]

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    bars_in = ax.bar(x - w / 2, in_vals, w, label="In-domain (HemoSet val)", color="tab:blue")
    bars_out = ax.bar(x + w / 2, out_vals, w, label="Out-of-domain (CholecSeg8k)", color="tab:orange")

    for i, (bi, bo, vi, vo) in enumerate(zip(bars_in, bars_out, in_vals, out_vals)):
        if vi > 1e-8:
            drop = (vi - vo) / vi * 100.0
        else:
            drop = 0.0
        ax.annotate(
            f"−{drop:.1f}%",
            xy=(bo.get_x() + bo.get_width() / 2, bo.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:orange",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.0, max(in_vals + out_vals) * 1.15))
    ax.legend()
    ax.set_title("Blood Segmentation Performance: In-Domain vs Out-of-Domain")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "performance_comparison.png", dpi=DPI)
    plt.close(fig)


def figure_failure_pie() -> None:
    r = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    fm = r.get("failure_modes", {})
    order = ["FALSE_NEGATIVE", "FALSE_POSITIVE", "PARTIAL", "CORRECT"]
    labels = order
    sizes = [int(fm.get(k, 0)) for k in order]
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]
    # Drop zero slices for pie readability; note zeros in subtitle
    pairs = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if not pairs:
        raise ValueError("No failure mode counts in training_results.json")
    labels_p = [p[0] for p in pairs]
    sizes_p = [p[1] for p in pairs]
    colors_p = [p[2] for p in pairs]
    total = sum(sizes)
    explode = [0.02] * len(sizes_p)

    _set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes_p,
        labels=labels_p,
        colors=colors_p,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100.0 * total))})",
        explode=explode,
        startangle=90,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Failure Mode Distribution on Out-of-Domain Data")
    fn_n = int(fm.get("FALSE_NEGATIVE", 0))
    if fn_n == 0:
        sub = (
            "False positives dominate when no blood is present; partial overlap on all blood frames. "
            "FALSE_NEGATIVE count: 0 (model always predicted some positive pixels on blood frames)."
        )
    else:
        sub = "Missed blood (False Negative) is the most dangerous failure mode"
    fig.suptitle(sub, fontsize=10, y=0.02)
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(FIGURES_DIR / "failure_mode_breakdown.png", dpi=DPI)
    plt.close(fig)


def figure_correlation() -> None:
    rows: list[dict[str, str]] = []
    with CORRELATION_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: abs(float(r["spearman_r"])), reverse=True)

    labels_map = {
        "mean_brightness": "Brightness",
        "mean_saturation": "Saturation",
        "mean_R": "R channel",
        "mean_G": "G channel",
        "mean_B": "B channel",
        "blood_coverage_pct": "Blood coverage % (GT)",
    }

    feats = [r["feature"] for r in rows]
    rhos = [float(r["spearman_r"]) for r in rows]
    ps = [float(r["p_value"]) for r in rows]
    y_labels = [labels_map.get(f, f) for f in feats]
    colors = ["#27ae60" if rho >= 0 else "#c0392b" for rho in rhos]

    _set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(rhos))
    ax.barh(y_pos, rhos, color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Spearman ρ (vs Dice on blood frames)")
    for i, (rho, p) in enumerate(zip(rhos, ps)):
        p_txt = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        ax.annotate(
            p_txt,
            xy=(rho, i),
            xytext=(3 if rho >= 0 else -3, 0),
            textcoords="offset points",
            ha="left" if rho >= 0 else "right",
            va="center",
            fontsize=7,
        )
    ax.set_title("Which Visual Properties Predict Segmentation Failure?")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "visual_correlation.png", dpi=DPI)
    plt.close(fig)


def _overlay_blood(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """mask float (H,W) 0–1."""
    out = img_rgb.astype(np.float32)
    red = np.array([255.0, 0.0, 0.0])
    m = (mask > 0.5).astype(np.float32)[..., None]
    blended = out * (1 - alpha * m) + red * alpha * m
    return np.clip(blended, 0, 255).astype(np.uint8)


@torch.no_grad()
def _predict_mask(model: torch.nn.Module, device: torch.device, img_rgb: np.ndarray) -> np.ndarray:
    """Return float mask H,W in [0,1] at 256²."""
    h, w = img_rgb.shape[:2]
    dummy = np.zeros((h, w), dtype=np.float32)
    img_r, _ = resize_both(img_rgb, dummy)
    x = normalize_imagenet(img_r)
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    pred = model(t).cpu().numpy()[0, 0]
    return pred.astype(np.float32)


def figure_qualitative() -> None:
    if not BEST_CKPT.is_file():
        raise FileNotFoundError(f"Missing {BEST_CKPT}")

    rows: list[dict[str, Any]] = []
    with PER_FRAME_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    correct = [r for r in rows if r.get("failure_mode") == "CORRECT" and float(r["dice"]) >= 0.5]
    partial = [r for r in rows if r.get("failure_mode") == "PARTIAL"]
    fp_pool = [r for r in rows if r.get("failure_mode") == "FALSE_POSITIVE"]
    fn_pool = [r for r in rows if r.get("failure_mode") == "FALSE_NEGATIVE"]

    random.seed(RNG_SEED)
    rows_show: list[tuple[dict[str, Any], str]] = []

    c_sorted = sorted(correct, key=lambda x: float(x["dice"]), reverse=True)
    for r in c_sorted[:2]:
        rows_show.append((r, "CORRECT"))

    if len(fn_pool) >= 2:
        for r in random.sample(fn_pool, 2):
            rows_show.append((r, "FALSE_NEGATIVE"))
    else:
        for r in sorted(partial, key=lambda x: float(x["dice"]))[:2]:
            rows_show.append((r, "PARTIAL (worst overlap)"))

    fp_pick = random.sample(fp_pool, min(2, len(fp_pool))) if fp_pool else []
    for r in fp_pick:
        rows_show.append((r, "FALSE_POSITIVE"))

    while len(rows_show) < 6 and fp_pool:
        rows_show.append((fp_pool[len(rows_show) % len(fp_pool)], "FALSE_POSITIVE"))
    rows_show = rows_show[:6]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model = build_unet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _set_style()
    nrows = len(rows_show)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, max(4, 1.3 * nrows)))
    if nrows == 1:
        axes = np.asarray(axes).reshape(1, -1)

    for ri, (row, mode_label) in enumerate(rows_show):
        rel = row["frame_id"]
        fp = DATA_CHOLEC / rel
        img_rgb, gt_mask = load_cholec_image_mask(fp)
        img_r, gt_r = resize_both(img_rgb, gt_mask)
        pred = _predict_mask(model, device, img_rgb)

        o_gt = _overlay_blood(img_r, gt_r)
        o_pr = _overlay_blood(img_r, pred)

        dice = float(row["dice"])
        for ci, (im, title) in enumerate(
            zip(
                [img_r, o_gt, o_pr],
                ["Image", "GT (blood = red)", "Prediction (blood = red)"],
            )
        ):
            ax = axes[ri, ci]
            ax.imshow(im)
            ax.axis("off")
            if ri == 0:
                ax.set_title(title, fontsize=10)
        axes[ri, 0].set_ylabel(f"{mode_label}\nDice={dice:.3f}", fontsize=9)

    fig.suptitle("Qualitative Results on Out-of-Domain Laparoscopic Data", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "qualitative_examples.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_phase4() -> None:
    _ensure_dirs()
    _set_style()
    print("Figure 1: domain gap...", flush=True)
    figure_domain_gap()
    print("Figure 2: performance bars...", flush=True)
    figure_performance()
    print("Figure 3: failure pie...", flush=True)
    figure_failure_pie()
    print("Figure 4: correlations...", flush=True)
    figure_correlation()
    print("Figure 5: qualitative grid...", flush=True)
    figure_qualitative()
    print(f"Saved 5 figures under {FIGURES_DIR}", flush=True)
    print("PHASE 4 COMPLETE", flush=True)


def main() -> None:
    run_phase4()


if __name__ == "__main__":
    main()
