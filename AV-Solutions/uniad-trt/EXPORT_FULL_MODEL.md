# UniAD Full Model: Export & Deploy Workflow

## One-Shot (after training completes)

```bash
cd /localhome/local-samehm/DL4AGX/AV-Solutions/uniad-trt
./export_full_model_to_trt.sh 20 all    # epoch 20, both FP32 and FP16
```

## Manual Step-by-Step

### Container Setup (one-time)

```bash
# Training container (already running)
docker run -d --gpus all --name uniad_training --shm-size=32g \
  -v /localhome/local-samehm/DL4AGX/AV-Solutions/uniad-trt:/workspace/uniad-trt \
  uniad_torch1.12 sleep infinity

# TRT container
docker run -d --gpus all --name uniad_trt --shm-size=8g \
  -v /localhome/local-samehm/DL4AGX/AV-Solutions:/workspace/AV-Solutions \
  nvcr.io/nvidia/tensorrt:24.12-py3 sleep infinity

# Build TRT plugins (one-time, inside uniad_trt)
docker exec uniad_trt bash -c "
  cd /workspace/AV-Solutions/uniad-trt/inference_app/enqueueV3 &&
  mkdir -p build && cd build &&
  cmake .. -DTENSORRT_PATH=/usr/lib/x86_64-linux-gnu -DTARGET_GPU_SM=86 &&
  make -j\$(nproc)
"
```

### Step 1: Copy checkpoint

```bash
docker exec uniad_training bash -c "
  cd /workspace/uniad-trt &&
  mkdir -p UniAD/ckpts &&
  cp UniAD_train/UniAD/projects/work_dirs/stage2_e2e/full_e2e/epoch_20.pth \
     UniAD/ckpts/full_e2e_ep20.pth
"
```

### Step 2: Data symlink (one-time)

```bash
docker exec uniad_training bash -c "
  cd /workspace/uniad-trt/UniAD &&
  ln -sf /workspace/uniad-trt/UniAD_train/UniAD/data ./data
"
```

### Step 3: Generate full-model input data (one-time)

The pre-shipped `nuscenes_np/` data is for the tiny model (50x50 BEV).
The full model needs 200x200. Regenerate:

```bash
docker exec uniad_training bash -c "
  cd /workspace/uniad-trt/UniAD &&
  PYTHONPATH=/workspace/uniad-trt/UniAD:\$PYTHONPATH \
  CUDA_VISIBLE_DEVICES='' \
  python3 tools/process_metadata.py \
    --config ./projects/configs/stage2_e2e/full_e2e_trt_p.py \
    --dump_folder ./nuscenes_np \
    --dump_trt_path ./nuscenes_np/uniad_trt_input \
    --dump_onnx_path ./nuscenes_np/uniad_onnx_input \
    --num_frame 69
"
```

### Step 4: Export to ONNX (~16 min)

```bash
docker exec uniad_training bash -c "
  cd /workspace/uniad-trt/UniAD &&
  CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh \
    ./projects/configs/stage2_e2e/full_e2e_trt_p.py \
    ./ckpts/full_e2e_ep20.pth 1
"
# Output: UniAD/onnx/uniad_full_cp.repaired.onnx
```

### Step 5: Build TRT engine (~5 min)

```bash
docker exec uniad_trt bash -c "
  cd /workspace/AV-Solutions/uniad-trt/inference_app/enqueueV3 &&

  SHAPES='prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901'

  SHAPES_MAX='prev_track_intances0:1150x512,prev_track_intances1:1150x3,prev_track_intances3:1150,prev_track_intances4:1150,prev_track_intances5:1150,prev_track_intances6:1150,prev_track_intances8:1150,prev_track_intances9:1150x10,prev_track_intances11:1150x4x256,prev_track_intances12:1150x4,prev_track_intances13:1150'

  # FP32
  trtexec \
    --onnx=/workspace/AV-Solutions/uniad-trt/UniAD/onnx/uniad_full_cp.repaired.onnx \
    --saveEngine=/workspace/AV-Solutions/uniad-trt/UniAD/onnx/uniad_full_ep20_fp32.engine \
    --staticPlugins=./build/libuniad_plugin.so \
    --profilingVerbosity=detailed \
    --tacticSources=+CUBLAS \
    --minShapes=\$SHAPES --optShapes=\$SHAPES --maxShapes=\$SHAPES_MAX \
    --skipInference

  # FP16 (add --fp16)
  trtexec \
    --onnx=/workspace/AV-Solutions/uniad-trt/UniAD/onnx/uniad_full_cp.repaired.onnx \
    --saveEngine=/workspace/AV-Solutions/uniad-trt/UniAD/onnx/uniad_full_ep20_fp16.engine \
    --staticPlugins=./build/libuniad_plugin.so \
    --profilingVerbosity=detailed \
    --tacticSources=+CUBLAS \
    --fp16 \
    --minShapes=\$SHAPES --optShapes=\$SHAPES --maxShapes=\$SHAPES_MAX \
    --skipInference
"
```

### Step 6: Run inference

```bash
docker exec uniad_trt bash -c "
  cd /workspace/AV-Solutions/uniad-trt/inference_app/enqueueV3 &&
  ln -sf /workspace/AV-Solutions/uniad-trt/UniAD/data ./data &&
  LD_LIBRARY_PATH=\$LD_LIBRARY_PATH ./build/uniad \
    /workspace/AV-Solutions/uniad-trt/UniAD/onnx/uniad_full_ep20_fp32.engine \
    ./build/libuniad_plugin.so \
    /workspace/AV-Solutions/uniad-trt/UniAD/nuscenes_np/uniad_trt_input \
    ./output \
    69
"
```

## Key Gotchas

### 1. BEV Resolution Mismatch (ONNX export fails)

The pre-shipped `nuscenes_np/uniad_onnx_input/` has data at **50x50** (tiny model).
The full model needs **200x200**. If you see this error during ONNX export:

```
RuntimeError: The size of tensor a (40000) must match the size of tensor b (2500)
```

It means Step 3 was skipped. Regenerate the input data with `process_metadata.py`.

### 2. C++ Inference App Hardcoded for Tiny Model

The original `inference_app/enqueueV3/` code was written for the tiny model.
Two files needed changes for the full model:

**`include/uniad.hpp`** — Buffer shapes updated:
- `prev_bev` / `bev_embed`: `{2500, 1, 256}` → `{40000, 1, 256}` (200x200 BEV)
- `img`: `{1, 6, 3, 256, 416}` → `{1, 6, 3, 928, 1600}`
- `seg_out`: `{1, 5, 1, 50, 50}` → `{1, 5, 1, 200, 200}`

**`src/main.cpp`** — Image preprocessing resize scale:
- `resize_scale`: `0.25` → `1.0` (tiny: 1600x900 → 416x256, full: 1600x900 → 1600x928)

These changes are already applied in this repo. To switch back to tiny model,
revert the resize_scale to 0.25 and dimensions to the smaller values.

### 3. Rebuild After Header Changes

If you edit `uniad.hpp`, you must do a **clean rebuild** (`make clean && make`)
because `make` alone may only rebuild the executable, not the shared libraries.
