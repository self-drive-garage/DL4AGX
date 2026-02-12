#!/usr/bin/env python3
"""Plot UniAD Stage 2 training loss curves from log file."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

LOG_FILE = "/localhome/local-samehm/DL4AGX/AV-Solutions/uniad-trt/UniAD_train/UniAD/train_stage2.log"
OUTPUT_FILE = "/localhome/local-samehm/DL4AGX/stage2_loss_curves.png"


def parse_log(filepath):
    pattern = re.compile(r"Epoch \[(\d+)\]\[(\d+)/(\d+)\]")
    records = []
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            iteration = int(m.group(2))
            total_iters = int(m.group(3))

            lr_match = re.search(r"lr: ([\d.e+-]+)", line)
            lr = float(lr_match.group(1)) if lr_match else None

            metrics = {}
            for km in re.finditer(r"([\w.]+): ([-\d.]+(?:e[+-]?\d+)?)", line):
                try:
                    metrics[km.group(1)] = float(km.group(2))
                except ValueError:
                    pass

            # Global step: (epoch-1)*total_iters + iteration
            global_step = (epoch - 1) * total_iters + iteration

            records.append({
                "epoch": epoch,
                "iter": iteration,
                "total_iters": total_iters,
                "global_step": global_step,
                "lr": lr,
                "metrics": metrics,
            })
    return records


def smooth(values, weight=0.95):
    """Exponential moving average for smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        s = weight * last + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed


def epoch_averages(records, key):
    """Compute per-epoch averages for a metric."""
    epoch_data = defaultdict(list)
    for r in records:
        if key in r["metrics"]:
            epoch_data[r["epoch"]].append(r["metrics"][key])
    epochs = sorted(epoch_data.keys())
    avgs = [np.mean(epoch_data[e]) for e in epochs]
    return epochs, avgs


def main():
    print("Parsing log file...")
    records = parse_log(LOG_FILE)
    print(f"Found {len(records)} iterations. Generating plots...")

    steps = [r["global_step"] for r in records]
    total_iters = records[0]["total_iters"]

    # Epoch boundary ticks
    epochs_seen = sorted(set(r["epoch"] for r in records))
    epoch_starts = [(e - 1) * total_iters for e in epochs_seen]

    fig = plt.figure(figsize=(24, 28))
    fig.suptitle("UniAD Stage 2 Training Loss Curves", fontsize=20, fontweight='bold', y=0.995)

    # ── Plot 1: Total Loss ──────────────────────────────────────────────
    ax1 = fig.add_subplot(4, 2, 1)
    vals = [r["metrics"].get("loss", np.nan) for r in records]
    ax1.plot(steps, vals, alpha=0.15, color='C0', linewidth=0.5)
    ax1.plot(steps, smooth(vals, 0.97), color='C0', linewidth=2, label='Smoothed')
    ep_x, ep_y = epoch_averages(records, "loss")
    ep_steps = [(e - 0.5) * total_iters for e in ep_x]
    ax1.plot(ep_steps, ep_y, 'o-', color='red', markersize=5, linewidth=1.5, label='Epoch avg')
    ax1.set_ylabel("Total Loss", fontsize=12)
    ax1.set_title("Total Loss", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=40)

    # ── Plot 2: Motion Losses ───────────────────────────────────────────
    ax2 = fig.add_subplot(4, 2, 2)
    for key, label, color in [
        ("motion.loss_traj", "Traj Loss", "C0"),
        ("motion.min_ade", "Min ADE", "C1"),
        ("motion.min_fde", "Min FDE", "C2"),
        ("motion.mr", "Miss Rate", "C3"),
    ]:
        vals = [r["metrics"].get(key, np.nan) for r in records]
        ax2.plot(steps, smooth(vals, 0.97), color=color, linewidth=1.8, label=label)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Motion Prediction Losses", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Planning Losses ─────────────────────────────────────────
    ax3 = fig.add_subplot(4, 2, 3)
    for key, label, color in [
        ("planning.loss_ade", "ADE", "C0"),
        ("planning.loss_collision_0", "Collision T=0", "C1"),
        ("planning.loss_collision_1", "Collision T=1", "C2"),
        ("planning.loss_collision_2", "Collision T=2", "C3"),
    ]:
        vals = [r["metrics"].get(key, np.nan) for r in records]
        ax3.plot(steps, smooth(vals, 0.97), color=color, linewidth=1.8, label=label)
    ax3.set_ylabel("Loss", fontsize=12)
    ax3.set_title("Planning Losses", fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Map Losses ──────────────────────────────────────────────
    ax4 = fig.add_subplot(4, 2, 4)
    for key, label, color in [
        ("map.loss_cls", "Classification", "C0"),
        ("map.loss_bbox", "BBox", "C1"),
        ("map.loss_iou", "IoU", "C2"),
        ("map.loss_mask_things", "Mask Things", "C3"),
    ]:
        vals = [r["metrics"].get(key, np.nan) for r in records]
        ax4.plot(steps, smooth(vals, 0.97), color=color, linewidth=1.8, label=label)
    ax4.set_ylabel("Loss", fontsize=12)
    ax4.set_title("Map Losses", fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Occupancy Losses ────────────────────────────────────────
    ax5 = fig.add_subplot(4, 2, 5)
    for key, label, color in [
        ("occ.loss_dice", "Dice", "C0"),
        ("occ.loss_mask", "Mask", "C1"),
        ("occ.loss_aux_dice", "Aux Dice", "C2"),
        ("occ.loss_aux_mask", "Aux Mask", "C3"),
    ]:
        vals = [r["metrics"].get(key, np.nan) for r in records]
        ax5.plot(steps, smooth(vals, 0.97), color=color, linewidth=1.8, label=label)
    ax5.set_ylabel("Loss", fontsize=12)
    ax5.set_title("Occupancy Losses", fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: Tracking Losses (frame 0, last decoder layer) ──────────
    ax6 = fig.add_subplot(4, 2, 6)
    for key, label, color in [
        ("track.frame_0_loss_cls_5", "Classification", "C0"),
        ("track.frame_0_loss_bbox_5", "BBox", "C1"),
        ("track.frame_0_loss_past_trajs_5", "Past Trajs", "C2"),
    ]:
        vals = [r["metrics"].get(key, np.nan) for r in records]
        ax6.plot(steps, smooth(vals, 0.97), color=color, linewidth=1.8, label=label)
    ax6.set_ylabel("Loss", fontsize=12)
    ax6.set_title("Tracking Losses (Frame 0, Layer 5)", fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # ── Plot 7: Learning Rate ───────────────────────────────────────────
    ax7 = fig.add_subplot(4, 2, 7)
    lrs = [r["lr"] for r in records]
    ax7.plot(steps, lrs, color='C4', linewidth=2)
    ax7.set_ylabel("Learning Rate", fontsize=12)
    ax7.set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    # ── Plot 8: Gradient Norm ───────────────────────────────────────────
    ax8 = fig.add_subplot(4, 2, 8)
    vals = [r["metrics"].get("grad_norm", np.nan) for r in records]
    ax8.plot(steps, vals, alpha=0.15, color='C5', linewidth=0.5)
    ax8.plot(steps, smooth(vals, 0.97), color='C5', linewidth=2, label='Smoothed')
    ax8.set_ylabel("Gradient Norm", fontsize=12)
    ax8.set_title("Gradient Norm", fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    # Add epoch markers and shared x-label to all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        for es in epoch_starts[1:]:
            ax.axvline(x=es, color='gray', linestyle='--', alpha=0.25, linewidth=0.5)
        ax.set_xlabel("Training Step (iteration)", fontsize=10)
        # Add epoch numbers as secondary ticks at top
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks([(e - 0.5) * total_iters for e in epochs_seen[::2]])
        ax_top.set_xticklabels([f'E{e}' for e in epochs_seen[::2]], fontsize=8, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
