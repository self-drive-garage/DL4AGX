#!/usr/bin/env python3
"""Analyze UniAD Stage 2 training log to track loss progression."""

import re
import sys
from collections import defaultdict

LOG_FILE = "/localhome/local-samehm/DL4AGX/AV-Solutions/uniad-trt/UniAD_train/UniAD/train_stage2.log"

def parse_log(filepath):
    """Parse training log and extract per-iteration metrics."""
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

            # Extract lr
            lr_match = re.search(r"lr: ([\d.e+-]+)", line)
            lr = float(lr_match.group(1)) if lr_match else None

            # Extract all key=value pairs
            metrics = {}
            for km in re.finditer(r"([\w.]+): ([-\d.]+(?:e[+-]?\d+)?)", line):
                key = km.group(1)
                try:
                    val = float(km.group(2))
                    metrics[key] = val
                except ValueError:
                    pass

            records.append({
                "epoch": epoch,
                "iter": iteration,
                "total_iters": total_iters,
                "lr": lr,
                "metrics": metrics,
            })

    return records


def compute_epoch_stats(records):
    """Compute per-epoch average/min/max for key losses."""
    epoch_data = defaultdict(lambda: defaultdict(list))
    for rec in records:
        ep = rec["epoch"]
        for key, val in rec["metrics"].items():
            epoch_data[ep][key].append(val)
    return epoch_data


def print_separator(char="=", width=120):
    print(char * width)


def main():
    print("Parsing log file...")
    records = parse_log(LOG_FILE)
    print(f"Found {len(records)} training iterations across the log.\n")

    if not records:
        print("No training records found!")
        sys.exit(1)

    # Basic info
    first = records[0]
    last = records[-1]
    print_separator()
    print("TRAINING OVERVIEW")
    print_separator()
    print(f"  First record: Epoch [{first['epoch']}][{first['iter']}/{first['total_iters']}]")
    print(f"  Last record:  Epoch [{last['epoch']}][{last['iter']}/{last['total_iters']}]")
    print(f"  Iterations per epoch: {first['total_iters']}")
    print(f"  Total logged iterations: {len(records)}")
    epochs_seen = sorted(set(r["epoch"] for r in records))
    print(f"  Epochs seen: {epochs_seen}")
    print(f"  Learning rate range: {first['lr']:.6e} -> {last['lr']:.6e}")
    print()

    # Per-epoch statistics
    epoch_data = compute_epoch_stats(records)

    header_losses = ["loss", "motion.loss_traj", "motion.min_ade", "motion.mr",
                     "map.loss_cls", "map.loss_iou", "occ.loss_dice", "occ.loss_mask",
                     "planning.loss_ade", "planning.loss_collision_0",
                     "track.frame_0_loss_cls_5", "track.frame_0_loss_past_trajs_5",
                     "grad_norm"]

    short_names = {
        "loss": "TotalLoss",
        "motion.loss_traj": "Mot.Traj",
        "motion.min_ade": "Mot.ADE",
        "motion.mr": "Mot.MR",
        "map.loss_cls": "Map.Cls",
        "map.loss_iou": "Map.IoU",
        "occ.loss_dice": "Occ.Dice",
        "occ.loss_mask": "Occ.Mask",
        "planning.loss_ade": "Plan.ADE",
        "planning.loss_collision_0": "Plan.Col0",
        "track.frame_0_loss_cls_5": "Trk.Cls",
        "track.frame_0_loss_past_trajs_5": "Trk.Traj",
        "grad_norm": "GradNorm",
    }

    print_separator()
    print("PER-EPOCH AVERAGE LOSSES")
    print_separator()

    print(f"{'Epoch':>6} {'#Iters':>7}", end="")
    for loss_name in header_losses:
        sn = short_names.get(loss_name, loss_name[:10])
        print(f" {sn:>10}", end="")
    print()
    print("-" * 150)

    for ep in sorted(epoch_data.keys()):
        n_iters = len(epoch_data[ep].get("loss", []))
        print(f"{ep:>6} {n_iters:>7}", end="")
        for loss_name in header_losses:
            vals = epoch_data[ep].get(loss_name, [])
            if vals:
                avg = sum(vals) / len(vals)
                print(f" {avg:>10.4f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    print()

    # Total loss detail
    print_separator()
    print("TOTAL LOSS PROGRESSION (per-epoch avg, min, max, std)")
    print_separator()
    print(f"{'Epoch':>6} {'Avg':>10} {'Min':>10} {'Max':>10} {'Std':>10} {'#Iters':>7}")
    print("-" * 60)

    for ep in sorted(epoch_data.keys()):
        vals = epoch_data[ep].get("loss", [])
        if vals:
            avg = sum(vals) / len(vals)
            mn = min(vals)
            mx = max(vals)
            std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"{ep:>6} {avg:>10.2f} {mn:>10.2f} {mx:>10.2f} {std:>10.2f} {len(vals):>7}")

    print()

    # Within-epoch samples
    print_separator()
    print("WITHIN-EPOCH LOSS SAMPLES (first/last 3 iterations per epoch)")
    print_separator()

    for ep in sorted(epoch_data.keys()):
        ep_records = [r for r in records if r["epoch"] == ep]
        if len(ep_records) < 6:
            samples = ep_records
        else:
            samples = ep_records[:3] + ep_records[-3:]

        print(f"\n  Epoch {ep} ({len(ep_records)} iterations):")
        for rec in samples:
            loss_val = rec["metrics"].get("loss", float("nan"))
            mot_ade = rec["metrics"].get("motion.min_ade", float("nan"))
            plan_ade = rec["metrics"].get("planning.loss_ade", float("nan"))
            print(f"    Iter {rec['iter']:>5}/{rec['total_iters']}  loss={loss_val:>8.2f}  mot_ade={mot_ade:.4f}  plan_ade={plan_ade:.4f}")

    print()

    # Spike detection
    print_separator()
    print("LOSS SPIKE DETECTION (iterations where total loss > 2x epoch average)")
    print_separator()

    spike_count = 0
    for ep in sorted(epoch_data.keys()):
        vals = epoch_data[ep].get("loss", [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        ep_records = [r for r in records if r["epoch"] == ep]
        for rec in ep_records:
            loss_val = rec["metrics"].get("loss", 0)
            if loss_val > 2 * avg:
                print(f"  Epoch [{ep}][{rec['iter']}/{rec['total_iters']}]: loss={loss_val:.2f} (epoch avg={avg:.2f}, ratio={loss_val/avg:.2f}x)")
                spike_count += 1

    if spike_count == 0:
        print("  No significant spikes detected.")
    else:
        print(f"\n  Total spikes: {spike_count}")

    print()

    # Learning rate schedule
    print_separator()
    print("LEARNING RATE SCHEDULE")
    print_separator()
    for ep in sorted(epoch_data.keys()):
        ep_records = [r for r in records if r["epoch"] == ep]
        if ep_records:
            lr_first = ep_records[0]["lr"]
            lr_last = ep_records[-1]["lr"]
            print(f"  Epoch {ep:>3}: lr start={lr_first:.6e}, lr end={lr_last:.6e}")

    print()

    # Overall summary
    print_separator()
    print("SUMMARY & ASSESSMENT")
    print_separator()

    first_ep = min(epoch_data.keys())
    last_ep = max(epoch_data.keys())
    first_epoch_losses = epoch_data[first_ep].get("loss", [])
    last_epoch_losses = epoch_data[last_ep].get("loss", [])

    if first_epoch_losses and last_epoch_losses:
        first_avg = sum(first_epoch_losses) / len(first_epoch_losses)
        last_avg = sum(last_epoch_losses) / len(last_epoch_losses)
        reduction_pct = (1 - last_avg / first_avg) * 100
        print(f"  Total loss: {first_avg:.2f} (epoch {first_ep} avg) -> {last_avg:.2f} (epoch {last_ep} avg)")
        print(f"  Overall reduction: {reduction_pct:.1f}%")

    # Check recent trend
    recent_epochs = sorted(epoch_data.keys())[-5:]
    if len(recent_epochs) >= 2:
        recent_avgs = []
        for ep in recent_epochs:
            vals = epoch_data[ep].get("loss", [])
            if vals:
                recent_avgs.append((ep, sum(vals) / len(vals)))

        print(f"\n  Recent epoch averages (total loss):")
        for ep, avg in recent_avgs:
            print(f"    Epoch {ep}: {avg:.2f}")

        if len(recent_avgs) >= 2:
            if recent_avgs[-1][1] < recent_avgs[0][1]:
                print(f"\n  -> Loss is STILL DECREASING in recent epochs.")
            elif recent_avgs[-1][1] > recent_avgs[0][1] * 1.05:
                print(f"\n  -> WARNING: Loss has INCREASED in recent epochs!")
            else:
                print(f"\n  -> Loss has PLATEAUED in recent epochs.")

    # Component trends
    print(f"\n  Component loss trends (epoch {first_ep} avg -> epoch {last_ep} avg):")
    component_losses = [
        ("motion.loss_traj", "Motion Trajectory"),
        ("motion.min_ade", "Motion ADE"),
        ("motion.min_fde", "Motion FDE"),
        ("motion.mr", "Motion Miss Rate"),
        ("map.loss_cls", "Map Classification"),
        ("map.loss_bbox", "Map BBox"),
        ("map.loss_iou", "Map IoU"),
        ("map.loss_mask_things", "Map Mask Things"),
        ("occ.loss_dice", "Occupancy Dice"),
        ("occ.loss_mask", "Occupancy Mask"),
        ("planning.loss_ade", "Planning ADE"),
        ("planning.loss_collision_0", "Planning Collision"),
        ("track.frame_0_loss_cls_5", "Track Classification"),
        ("track.frame_0_loss_bbox_5", "Track BBox"),
        ("track.frame_0_loss_past_trajs_5", "Track Past Trajs"),
    ]

    for loss_key, label in component_losses:
        first_vals = epoch_data[first_ep].get(loss_key, [])
        last_vals = epoch_data[last_ep].get(loss_key, [])
        if first_vals and last_vals:
            f_avg = sum(first_vals) / len(first_vals)
            l_avg = sum(last_vals) / len(last_vals)
            change = ((l_avg - f_avg) / abs(f_avg)) * 100 if f_avg != 0 else 0
            direction = "IMPROVED" if l_avg < f_avg else "WORSENED"
            print(f"    {label:<25s}: {f_avg:>10.4f} -> {l_avg:>10.4f}  ({change:>+8.1f}%) {direction}")

    print()


if __name__ == "__main__":
    main()
