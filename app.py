"""
Phase 5 — Streamlit dashboard: domain gap, failure analysis, qualitative browser.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.data_pipeline import DATA_CHOLEC, load_cholec_image_mask, resize_both
from src.models import build_unet
from src.train import BEST_CKPT
from src.visualize import _overlay_blood, _predict_mask

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_RESULTS = PROJECT_ROOT / "outputs" / "results"
FIGURES = PROJECT_ROOT / "outputs" / "figures"

DOMAIN_STATS = OUT_RESULTS / "domain_stats.json"
TRAINING_RESULTS = OUT_RESULTS / "training_results.json"
PER_FRAME = OUT_RESULTS / "per_frame_analysis.csv"
CORRELATION = OUT_RESULTS / "correlation_analysis.csv"


def _dark_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        [data-testid="stSidebar"] { background-color: #161b22; }
        h1, h2, h3 { color: #f0f6fc !important; }
        .delta-high { color: #f85149; font-weight: 600; }
        .delta-low { color: #3fb950; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_segmentation_model():
    if not BEST_CKPT.is_file():
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model = build_unet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


@st.cache_data
def load_domain_stats() -> dict:
    return json.loads(DOMAIN_STATS.read_text(encoding="utf-8"))


@st.cache_data
def load_training_results() -> dict:
    return json.loads(TRAINING_RESULTS.read_text(encoding="utf-8"))


@st.cache_data
def load_per_frame() -> pd.DataFrame:
    return pd.read_csv(PER_FRAME)


@st.cache_data
def load_correlation() -> pd.DataFrame:
    return pd.read_csv(CORRELATION)


def _delta_color_class(abs_diff: float, max_diff: float) -> str:
    if max_diff < 1e-12:
        return "delta-low"
    t = abs_diff / max_diff
    return "delta-high" if t > 0.5 else "delta-low"


def page_domain() -> None:
    st.title("The Domain Gap")
    fig_path = FIGURES / "domain_gap_visual.png"
    if fig_path.is_file():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.warning(f"Missing figure: {fig_path}. Run `python -m src.visualize` first.")

    dom = load_domain_stats()
    tr = load_training_results()
    h = dom["hemoset_train"]
    c = dom["cholecseg8k_test"]

    metrics = [
        ("mean_R", "Mean R"),
        ("mean_G", "Mean G"),
        ("mean_B", "Mean B"),
        ("mean_brightness", "Mean brightness"),
        ("mean_saturation", "Mean saturation"),
    ]
    diffs = {k: abs(float(h[k]) - float(c[k])) for k, _ in metrics}
    max_d = max(diffs.values()) if diffs else 1.0
    top_key = max(diffs, key=diffs.get) if diffs else "mean_brightness"
    top_label = dict(metrics).get(top_key, top_key)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("HemoSet (train)")
        for key, label in metrics:
            cls = _delta_color_class(diffs[key], max_d)
            st.markdown(
                f'<p class="{cls}">{label}: {float(h[key]):.4f}</p>',
                unsafe_allow_html=True,
            )
        st.write(f"**Training images:** {int(h['n_images'])}")
        st.write(f"**Mean blood coverage %:** {float(h['mean_blood_coverage_pct']):.2f}")
    with col2:
        st.subheader("CholecSeg8k (OOD test)")
        for key, label in metrics:
            cls = _delta_color_class(diffs[key], max_d)
            st.markdown(
                f'<p class="{cls}">{label}: {float(c[key]):.4f}</p>',
                unsafe_allow_html=True,
            )
        st.write(f"**Test images:** {int(c['n_images'])}")
        st.write(
            f"**Frames with blood:** {int(c['frames_with_blood'])}  ·  "
            f"**without blood:** {int(c['frames_without_blood'])}"
        )
        st.write(f"**Mean blood coverage %:** {float(c['mean_blood_coverage_pct']):.2f}")

    idice = float(tr["hemoset_val_dice"])
    odice = float(tr["cholecseg8k_dice"])
    st.warning(
        f"The two domains differ most in **{top_label}** (largest gap in mean R/G/B, brightness, or saturation). "
        f"This gap is part of why blood segmentation drops from **in-domain Dice {idice:.3f}** to "
        f"**out-of-domain Dice {odice:.3f}** — alongside a failure mode dominated by **false positives** on laparoscopic frames, not “missed darkness.”"
    )


def page_failure() -> None:
    st.title("Why Does the Model Fail?")
    c1, c2 = st.columns(2)
    pie = FIGURES / "failure_mode_breakdown.png"
    corr = FIGURES / "visual_correlation.png"
    with c1:
        if pie.is_file():
            st.image(str(pie), use_container_width=True)
        else:
            st.info("Run Phase 4 to generate failure_mode_breakdown.png")
    with c2:
        if corr.is_file():
            st.image(str(corr), use_container_width=True)
        else:
            st.info("Run Phase 4 to generate visual_correlation.png")

    df = load_per_frame()
    feat_options = {
        "mean_brightness": "Brightness",
        "mean_saturation": "Saturation",
        "mean_R": "R",
        "mean_G": "G",
        "mean_B": "B",
        "blood_coverage_pct": "Blood coverage % (GT)",
    }
    xfeat = st.selectbox("X-axis (visual feature)", list(feat_options.keys()), format_func=lambda k: feat_options[k])

    color_map = {
        "FALSE_NEGATIVE": "#c0392b",
        "FALSE_POSITIVE": "#e67e22",
        "PARTIAL": "#f1c40f",
        "CORRECT": "#27ae60",
    }
    fig_sc = go.Figure()
    for mode, g in df.groupby("failure_mode"):
        x = g[xfeat].astype(float)
        y = g["dice"].astype(float)
        fig_sc.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=mode,
                marker=dict(size=5, color=color_map.get(mode, "#888")),
                opacity=0.65,
            )
        )
    xd = df[xfeat].astype(float).values
    yd = df["dice"].astype(float).values
    mask = np.isfinite(xd) & np.isfinite(yd)
    if mask.sum() > 2:
        try:
            coef = np.polyfit(xd[mask], yd[mask], 1)
            xs = np.linspace(float(np.min(xd[mask])), float(np.max(xd[mask])), 50)
            fig_sc.add_trace(
                go.Scatter(
                    x=xs,
                    y=coef[0] * xs + coef[1],
                    mode="lines",
                    name="Trend (linear fit)",
                    line=dict(color="white", dash="dash"),
                )
            )
        except (np.linalg.LinAlgError, ValueError):
            pass
    fig_sc.update_layout(
        template="plotly_dark",
        xaxis_title=feat_options[xfeat],
        yaxis_title="Dice",
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Per-frame table")
    mode_filter = st.multiselect(
        "Filter by failure mode",
        sorted(df["failure_mode"].unique()),
        default=sorted(df["failure_mode"].unique()),
    )
    sub = df[df["failure_mode"].isin(mode_filter)].copy()
    sub = sub.sort_values("dice", ascending=True)
    st.caption("Sorted by Dice (ascending). Showing the **worst** frames first; adjust with slider below.")
    n_show = st.slider("Rows to show", 20, min(500, len(sub)), 20)
    st.dataframe(sub.head(n_show), use_container_width=True, height=320)


def page_qualitative() -> None:
    st.title("What the Model Sees")
    qfig = FIGURES / "qualitative_examples.png"
    if qfig.is_file():
        st.image(str(qfig), use_container_width=True)
    else:
        st.warning("Missing qualitative_examples.png — run Phase 4.")

    df = load_per_frame().reset_index(drop=True)
    corr_df = load_correlation()
    bright_row = corr_df[corr_df["feature"] == "mean_brightness"]
    p_bright = float(bright_row["p_value"].iloc[0]) if len(bright_row) else float("nan")
    bc_row = corr_df[corr_df["feature"] == "blood_coverage_pct"]
    rho_bc = float(bc_row["spearman_r"].iloc[0]) if len(bc_row) else float("nan")

    tr = load_training_results()
    fm = tr.get("failure_modes", {})
    n_fp = int(fm.get("FALSE_POSITIVE", 0))
    n_nb = int((df["has_blood"].astype(str).str.lower() == "false").sum())
    fp_rate = (n_fp / max(n_nb, 1)) * 100.0

    st.warning(
        f"Laparoscopic frames are **intentionally dark** in this dataset; low luminance is part of the domain gap (see Figure 1), not a display bug. "
        f"**Brightness** did not predict Dice on blood frames (Spearman p≈{p_bright:.2f}). "
        f"The strongest correlate was **GT blood coverage %** (ρ≈{rho_bc:.2f}): larger labeled bleeds associate with **lower** Dice here because predictions are spatially misaligned and FP-heavy. "
        f"False positives appeared on **~{fp_rate:.0f}%** of no-blood frames — a deployed system would alarm constantly, not miss bleeding silently."
    )

    model, device = load_segmentation_model()
    if model is None:
        st.error("Trained checkpoint not found. Train Phase 2 first.")
        return

    only_fn = st.checkbox("Show only FALSE_NEGATIVE frames", value=False)
    idx_df = df.copy()
    if only_fn:
        idx_df = idx_df[idx_df["failure_mode"] == "FALSE_NEGATIVE"]
        if len(idx_df) == 0:
            st.info("There are **no** FALSE_NEGATIVE frames in this run — the model always produced some positive pixels on blood frames. Uncheck the box to browse all frames.")
            idx_df = df.copy()

    idx_df = idx_df.reset_index(drop=True)
    max_i = max(0, len(idx_df) - 1)
    i = st.slider("Frame index (within current filter)", 0, max_i, 0)
    row = idx_df.iloc[i]
    rel = row["frame_id"]
    fp = DATA_CHOLEC / rel

    img_rgb, gt_mask = load_cholec_image_mask(fp)
    img_r, gt_r = resize_both(img_rgb, gt_mask)
    pred = _predict_mask(model, device, img_rgb)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Original (256²)")
        st.image(img_r, use_container_width=True)
    with c2:
        st.caption("Ground truth (red = blood)")
        st.image(_overlay_blood(img_r, gt_r), use_container_width=True)
    with c3:
        st.caption("Prediction (red = blood)")
        st.image(_overlay_blood(img_r, pred), use_container_width=True)

    st.json(
        {
            "frame_id": rel,
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "failure_mode": row["failure_mode"],
            "mean_brightness": float(row["mean_brightness"]),
            "mean_saturation": float(row["mean_saturation"]),
            "blood_coverage_pct": float(row["blood_coverage_pct"]),
            "has_blood": str(row["has_blood"]),
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="Surgical Blood Generalization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _dark_css()

    tr = load_training_results() if TRAINING_RESULTS.is_file() else {}
    with st.sidebar:
        st.title("Surgical Blood Generalization")
        st.caption(
            "A blood segmentation model trained on robotic surgery, tested on laparoscopic surgery."
        )
        st.divider()
        st.metric("In-domain Dice", f"{float(tr.get('hemoset_val_dice', 0)):.4f}")
        st.metric("Out-of-domain Dice", f"{float(tr.get('cholecseg8k_dice', 0)):.4f}")
        st.metric("Performance drop (Dice %)", f"{float(tr.get('dice_drop_pct', 0)):.2f}")
        st.divider()
        st.markdown("[GitHub: surgical-blood-generalization](https://github.com/youssof20/surgical-blood-generalization)")
        st.caption("Research only. Not for clinical use.")
        st.divider()
        page = st.radio(
            "Pages",
            ("Domain Explorer", "Failure Analysis", "Qualitative Results"),
            label_visibility="collapsed",
        )

    if page == "Domain Explorer":
        page_domain()
    elif page == "Failure Analysis":
        page_failure()
    else:
        page_qualitative()


if __name__ == "__main__":
    main()
