"""
Plot RL learning curves from TensorBoard event files.

Reads local TensorBoard logs and produces publication-quality figures
(seaborn + matplotlib) comparing multiple runs (e.g., dense reward vs sparse).

Usage examples:

  # Compare two runs
  python robometer/rl/plot_rl.py \
      --run-dirs runs/libero_90_t28__sac_libero__42__... \
                 runs/libero_90_t28__sparse__42__... \
      --labels "BT+MaxEnt (Ours)" "Sparse Reward" \
      --output figures/rl_success_rate.pdf

  # Auto-discover all runs under runs/
  python robometer/rl/plot_rl.py --run-root runs/ --output figures/rl_curves.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# TensorBoard reader
# ---------------------------------------------------------------------------

def read_tb_scalars(log_dir: str, tag: str) -> pd.DataFrame:
    """Read a scalar tag from a TensorBoard event file into a DataFrame."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return pd.DataFrame(columns=["step", "value"])
    events = ea.Scalars(tag)
    return pd.DataFrame({"step": [e.step for e in events],
                         "value": [e.value for e in events]})


def read_multiple_runs(
    run_dirs: list[str],
    tag: str,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Read the same scalar tag from multiple run directories."""
    frames = []
    for i, d in enumerate(run_dirs):
        df = read_tb_scalars(d, tag)
        if df.empty:
            print(f"[warn] tag '{tag}' not found in {d}")
            continue
        df["run"] = labels[i] if labels else os.path.basename(d)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Smoothing (EMA, same as TensorBoard / SAC-FLOW paper_plot.py)
# ---------------------------------------------------------------------------

def smooth(scalars: list[float], weight: float = 0.6) -> list[float]:
    """Exponential moving average. weight in [0, 1), higher = smoother."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        s = last * weight + (1 - weight) * point
        smoothed.append(s)
        last = s
    return smoothed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {
    "BT+MaxEnt (Ours)": "#2196F3",
    "Sparse Reward": "#FF9800",
}

ZH_LABELS = {
    "eval/success_rate": ("训练步数", "评估成功率"),
    "eval/mean_return": ("训练步数", "平均回合奖励"),
    "train/episode_return": ("训练步数", "训练回合奖励"),
    "train/episode_success": ("训练步数", "训练成功率"),
}

EN_LABELS = {
    "eval/success_rate": ("Training Steps", "Eval Success Rate"),
    "eval/mean_return": ("Training Steps", "Mean Episode Return"),
    "train/episode_return": ("Training Steps", "Episode Return"),
    "train/episode_success": ("Training Steps", "Episode Success"),
}


def set_pub_style(font_size: int = 14):
    """Set publication-quality matplotlib defaults."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.figsize": (7, 4.5),
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def plot_learning_curve(
    df: pd.DataFrame,
    tag: str,
    output_path: str,
    title: str | None = None,
    smooth_weight: float = 0.0,
    lang: str = "zh",
    ylim: tuple[float, float] | None = None,
):
    """
    Plot a learning-curve figure from a DataFrame with columns [step, value, run].

    If multiple entries exist per (step, run), seaborn draws mean +/- std shading
    (useful for multi-seed). For single seed, it draws a plain line.
    """
    set_pub_style()
    labels_map = ZH_LABELS if lang == "zh" else EN_LABELS
    xlabel, ylabel = labels_map.get(tag, ("Steps", "Value"))

    if smooth_weight > 0:
        smoothed = []
        for name, grp in df.groupby("run", sort=False):
            grp = grp.sort_values("step").copy()
            grp["value"] = smooth(grp["value"].tolist(), smooth_weight)
            smoothed.append(grp)
        df = pd.concat(smoothed, ignore_index=True)

    fig, ax = plt.subplots()
    run_names = df["run"].unique()

    palette = [PALETTE.get(r, None) for r in run_names]
    palette = palette if all(palette) else None

    sns.lineplot(
        data=df, x="step", y="value", hue="run",
        palette=palette, ax=ax, errorbar="sd",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.legend(title=None)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[plot] saved {output_path}")


def plot_all_metrics(
    run_dirs: list[str],
    labels: list[str],
    output_dir: str,
    smooth_weight: float = 0.6,
    lang: str = "zh",
):
    """Generate all standard RL figures for a set of runs."""
    configs = [
        ("eval/success_rate", "rl_success_rate", (0.0, 1.05)),
        ("eval/mean_return", "rl_eval_return", None),
        ("train/episode_return", "rl_train_return", None),
    ]
    for tag, fname, ylim in configs:
        df = read_multiple_runs(run_dirs, tag, labels)
        if df.empty:
            print(f"[skip] no data for {tag}")
            continue
        for ext in ("pdf", "png"):
            out = os.path.join(output_dir, f"{fname}.{ext}")
            plot_learning_curve(
                df, tag, out,
                smooth_weight=smooth_weight,
                lang=lang,
                ylim=ylim,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot RL learning curves from TensorBoard logs")
    p.add_argument("--run-dirs", nargs="+", help="Paths to TensorBoard run directories")
    p.add_argument("--labels", nargs="+", help="Legend labels (same order as --run-dirs)")
    p.add_argument("--run-root", type=str, help="Auto-discover all runs under this directory")
    p.add_argument("--tag", type=str, default=None,
                   help="Single scalar tag to plot (if omitted, plots all standard metrics)")
    p.add_argument("--output", type=str, default="figures/rl_success_rate.pdf",
                   help="Output file (used with --tag) or output directory (used without --tag)")
    p.add_argument("--smooth", type=float, default=0.6, help="EMA smoothing weight [0,1)")
    p.add_argument("--lang", choices=["zh", "en"], default="zh", help="Axis label language")
    return p.parse_args()


def discover_runs(root: str, label_pattern: str | None = None) -> tuple[list[str], list[str]]:
    """Auto-discover run directories under root, grouping by run name prefix."""
    dirs, labels = [], []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        has_events = any(f.startswith("events.out.tfevents") for f in os.listdir(full))
        if has_events:
            dirs.append(full)
            if "sparse" in name.lower():
                labels.append("Sparse Reward")
            else:
                labels.append("BT+MaxEnt (Ours)")
    return dirs, labels


def main():
    args = parse_args()

    if args.run_root:
        run_dirs, labels = discover_runs(args.run_root)
    elif args.run_dirs:
        run_dirs = args.run_dirs
        labels = args.labels or [os.path.basename(d) for d in run_dirs]
    else:
        print("Error: provide --run-dirs or --run-root")
        sys.exit(1)

    if not run_dirs:
        print("Error: no run directories found")
        sys.exit(1)

    print(f"[plot] found {len(run_dirs)} run(s):")
    for d, l in zip(run_dirs, labels):
        print(f"  {l}: {d}")

    if args.tag:
        df = read_multiple_runs(run_dirs, args.tag, labels)
        if df.empty:
            print(f"Error: no data for tag '{args.tag}'")
            sys.exit(1)
        plot_learning_curve(df, args.tag, args.output, smooth_weight=args.smooth, lang=args.lang)
    else:
        out_dir = os.path.splitext(args.output)[0] if "." in os.path.basename(args.output) else args.output
        plot_all_metrics(run_dirs, labels, out_dir, smooth_weight=args.smooth, lang=args.lang)


if __name__ == "__main__":
    main()
