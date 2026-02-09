## Stage 1 Training Loss Analysis

**Setup**: 8x NVIDIA A40 GPUs, PyTorch 1.12.1, CUDA 11.8, gloo backend, 6 epochs total, 3517 iterations/epoch, CosineAnnealing LR schedule (peak lr=2e-4).

**Training Progress**: Currently in **Epoch 4, iter 540/3517** (~60% complete overall).

### Total Loss Progression

| Point | Total Loss |
|---|---|
| Epoch 1, iter 10 (start) | **105.67** |
| Epoch 1, iter 100 | **81.68** |
| Epoch 1, iter 610 (lr warmup done) | **66.16** |
| Epoch 1, iter 3510 (end) | **55.46** |
| Epoch 2, iter 10 | **56.70** |
| Epoch 3, iter 10 | **53.25** |
| Epoch 4, latest (~iter 230-540) | **~47-50** |

### Key Observations

1. **Healthy convergence**: The total loss dropped from ~106 to ~48, a ~55% reduction. The steepest drop was in early Epoch 1 during warmup, which is expected.

2. **Tracking losses** (cls + bbox across 3 frames, 6 decoder layers):
   - Classification loss: 1.24 -> ~0.67-0.70 (good steady decrease)
   - BBox loss: 2.35 -> ~0.85-0.95 (strong improvement, the model is learning to localize objects)
   - Past trajectory loss: remains 0.0 throughout (this is expected for stage 1 - trajectory prediction is not active)

3. **Map losses**:
   - `map.loss_cls`: 1.21 -> ~0.68 (improving)
   - `map.loss_bbox`: 1.69 -> ~0.65 (significant improvement)
   - `map.loss_iou`: 2.03 -> ~1.45 (steady decrease)
   - `map.loss_mask_things`: 1.86 -> ~1.58 (slower but steady)
   - `map.loss_mask_stuff`: 0.06 -> ~0.03 (converged quickly, small magnitude)

4. **Gradient norm**: Stabilized around 50-60, no signs of gradient explosion or vanishing.

5. **Learning rate**: Peaked at 2e-4, now at ~1e-4 (cosine decay in epoch 4 of 6). The decay is helping with fine-grained convergence.

6. **Loss plateau**: The loss has slowed its descent in epochs 3-4 (~53 to ~48), which is typical as the model converges. The remaining 2 epochs with decaying LR should squeeze out a bit more.

**Overall**: Training looks healthy. No NaN/Inf issues, no loss spikes, steady monotonic convergence across all loss components. The gloo backend is working fine.

---

## Stage 2 Command

The stage 2 config is at `projects/configs/stage2_e2e/full_e2e.py`. It expects the stage 1 checkpoint at `ckpts/full_track_map.pth` (via `load_from`). Stage 2 runs for **20 epochs** of end-to-end training.

After stage 1 finishes, you'll need to:

1. Copy/symlink the best stage 1 checkpoint to the expected path:
   ```bash
   cp ./projects/work_dirs/stage1_track_map/full_track_map/latest.pth ./ckpts/full_track_map.pth
   ```

2. Run stage 2 (adapted from RUN_COMMAND.sh with gloo backend):
   ```bash
   nohup bash -c 'PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/uniad-trt/UniAD_train/UniAD:$PYTHONPATH \
      python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=28599 \
      ./tools/train.py ./projects/configs/stage2_e2e/full_e2e.py \
      --launcher pytorch --deterministic \
      --work-dir ./projects/work_dirs/stage2_e2e/full_e2e/ \
      --cfg-options data.workers_per_gpu=4 log_config.interval=10 dist_params.backend=gloo' \
      > train_stage2.log 2>&1 &
   ```

The key changes from the stage 1 command:
- Config: `full_track_map.py` -> `full_e2e.py`
- Work dir: `stage1_track_map/full_track_map/` -> `stage2_e2e/full_e2e/`
- Log file: `train_stage1.log` -> `train_stage2.log`
- `dist_params.backend=gloo` is kept since NCCL didn't work in this environment
