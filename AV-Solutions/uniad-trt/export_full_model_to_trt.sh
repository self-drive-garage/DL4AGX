#!/usr/bin/env bash
# =============================================================================
# UniAD Full Model: Checkpoint → ONNX → TensorRT Engine Pipeline
# =============================================================================
#
# Exports a training checkpoint to a TensorRT engine for the full-size
# UniAD model (ResNet-101, 200x200 BEV, 1600x928 images).
#
# Prerequisites:
#   - Training container "uniad_training" running with workspace mounted at
#     /workspace/uniad-trt (image: uniad_torch1.12)
#   - TRT container "uniad_trt" running with workspace mounted at
#     /workspace/AV-Solutions (image: nvcr.io/nvidia/tensorrt:24.12-py3)
#   - Plugins already built in uniad_trt container at:
#     /workspace/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_plugin.so
#   - NuScenes data available at UniAD_train/UniAD/data/
#   - Full-model input data already generated in UniAD/nuscenes_np/uniad_onnx_input/
#     (200x200 BEV resolution, 928x1600 images)
#
# Usage:
#   ./export_full_model_to_trt.sh <epoch_number> [precision]
#
# Examples:
#   ./export_full_model_to_trt.sh 20          # Export epoch 20, FP32
#   ./export_full_model_to_trt.sh 20 fp16     # Export epoch 20, FP16
#   ./export_full_model_to_trt.sh 20 all      # Export epoch 20, both FP32 and FP16
#
# =============================================================================

set -euo pipefail

EPOCH=${1:?Usage: $0 <epoch_number> [precision: fp32|fp16|all]}
PRECISION=${2:-fp32}

# --- Paths (relative to uniad-trt/) ---
TRAIN_WORK_DIR="UniAD_train/UniAD/projects/work_dirs/stage2_e2e/full_e2e"
DEPLOY_DIR="UniAD"
CKPT_DIR="${DEPLOY_DIR}/ckpts"
ONNX_DIR="${DEPLOY_DIR}/onnx"
CONFIG="./projects/configs/stage2_e2e/full_e2e_trt_p.py"
PLUGIN_PATH="/workspace/AV-Solutions/uniad-trt/inference_app/enqueueV3/build/libuniad_plugin.so"

CKPT_SRC="${TRAIN_WORK_DIR}/epoch_${EPOCH}.pth"
CKPT_DST="${CKPT_DIR}/full_e2e_ep${EPOCH}.pth"
ONNX_FILE="${ONNX_DIR}/uniad_full_cp.repaired.onnx"
ENGINE_FP32="${ONNX_DIR}/uniad_full_ep${EPOCH}_fp32.engine"
ENGINE_FP16="${ONNX_DIR}/uniad_full_ep${EPOCH}_fp16.engine"

TRAINING_CONTAINER="uniad_training"
TRT_CONTAINER="uniad_trt"

echo "============================================================"
echo "  UniAD Full Model Export Pipeline"
echo "  Epoch: ${EPOCH}  |  Precision: ${PRECISION}"
echo "============================================================"

# =============================================================================
# Step 1: Copy checkpoint to deployment directory
# =============================================================================
echo ""
echo "[Step 1/5] Copying checkpoint epoch_${EPOCH}.pth to deployment dir..."

docker exec ${TRAINING_CONTAINER} bash -c "
  cd /workspace/uniad-trt &&
  mkdir -p ${CKPT_DIR} &&
  cp ${CKPT_SRC} ${CKPT_DST}
"
echo "  → ${CKPT_DST}"

# =============================================================================
# Step 2: Ensure data symlink exists
# =============================================================================
echo ""
echo "[Step 2/5] Ensuring data symlink..."

docker exec ${TRAINING_CONTAINER} bash -c "
  cd /workspace/uniad-trt/${DEPLOY_DIR} &&
  if [ ! -L data ] && [ ! -d data ]; then
    ln -sf /workspace/uniad-trt/UniAD_train/UniAD/data ./data
    echo '  → Created symlink: data -> UniAD_train/UniAD/data'
  else
    echo '  → Data symlink already exists'
  fi
"

# =============================================================================
# Step 3: Regenerate input data for full model (if not already done)
# =============================================================================
echo ""
echo "[Step 3/5] Checking/generating full-model input data (200x200 BEV)..."

docker exec ${TRAINING_CONTAINER} bash -c "
  cd /workspace/uniad-trt/${DEPLOY_DIR} &&
  if [ -f nuscenes_np/uniad_onnx_input/gt_segmentation/0.npy ]; then
    SHAPE=\$(python3 -c \"import numpy as np; print(np.load('nuscenes_np/uniad_onnx_input/gt_segmentation/0.npy').shape)\")
    if echo \"\$SHAPE\" | grep -q '200, 200'; then
      echo '  → Full-model input data already exists (200x200). Skipping.'
    else
      echo '  → Input data exists but wrong resolution. Regenerating...'
      mv nuscenes_np/uniad_onnx_input nuscenes_np/uniad_onnx_input_backup_\$(date +%s)
      mv nuscenes_np/uniad_trt_input nuscenes_np/uniad_trt_input_backup_\$(date +%s) 2>/dev/null || true
      PYTHONPATH=/workspace/uniad-trt/${DEPLOY_DIR}:\$PYTHONPATH \
      CUDA_VISIBLE_DEVICES='' \
      python3 tools/process_metadata.py \
        --config ${CONFIG} \
        --dump_folder ./nuscenes_np \
        --dump_trt_path ./nuscenes_np/uniad_trt_input \
        --dump_onnx_path ./nuscenes_np/uniad_onnx_input \
        --num_frame 69
      echo '  → Input data regenerated at 200x200 resolution.'
    fi
  else
    echo '  → No input data found. Generating...'
    PYTHONPATH=/workspace/uniad-trt/${DEPLOY_DIR}:\$PYTHONPATH \
    CUDA_VISIBLE_DEVICES='' \
    python3 tools/process_metadata.py \
      --config ${CONFIG} \
      --dump_folder ./nuscenes_np \
      --dump_trt_path ./nuscenes_np/uniad_trt_input \
      --dump_onnx_path ./nuscenes_np/uniad_onnx_input \
      --num_frame 69
    echo '  → Input data generated.'
  fi
"

# =============================================================================
# Step 4: Export PyTorch checkpoint to ONNX
# =============================================================================
echo ""
echo "[Step 4/5] Exporting to ONNX..."

docker exec ${TRAINING_CONTAINER} bash -c "
  cd /workspace/uniad-trt/${DEPLOY_DIR} &&
  CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh \
    ${CONFIG} \
    ./ckpts/full_e2e_ep${EPOCH}.pth \
    1
"
echo "  → ONNX exported: ${ONNX_FILE}"

# =============================================================================
# Step 5: Build TensorRT engine(s)
# =============================================================================
echo ""
echo "[Step 5/5] Building TensorRT engine(s)..."

# Dynamic shape spec for tracking instances
SHAPES="prev_track_intances0:MINx512,prev_track_intances1:MINx3,prev_track_intances3:MIN,prev_track_intances4:MIN,prev_track_intances5:MIN,prev_track_intances6:MIN,prev_track_intances8:MIN,prev_track_intances9:MINx10,prev_track_intances11:MINx4x256,prev_track_intances12:MINx4,prev_track_intances13:MIN"

build_engine() {
  local output_engine=$1
  local extra_flags=$2
  local label=$3

  echo "  Building ${label} engine..."
  docker exec ${TRT_CONTAINER} bash -c "
    cd /workspace/AV-Solutions/uniad-trt/inference_app/enqueueV3 &&
    trtexec \
      --onnx=/workspace/AV-Solutions/uniad-trt/${ONNX_FILE} \
      --saveEngine=/workspace/AV-Solutions/uniad-trt/${output_engine} \
      --staticPlugins=./build/libuniad_plugin.so \
      --profilingVerbosity=detailed \
      --tacticSources=+CUBLAS \
      --minShapes=${SHAPES//MIN/901} \
      --optShapes=${SHAPES//MIN/901} \
      --maxShapes=${SHAPES//MIN/1150} \
      --skipInference \
      ${extra_flags} \
      2>&1 | tail -5
  "
  echo "  → ${label} engine saved: ${output_engine}"
}

case ${PRECISION} in
  fp32)
    build_engine "${ENGINE_FP32}" "" "FP32"
    ;;
  fp16)
    build_engine "${ENGINE_FP16}" "--fp16" "FP16"
    ;;
  all)
    build_engine "${ENGINE_FP32}" "" "FP32"
    build_engine "${ENGINE_FP16}" "--fp16" "FP16"
    ;;
  *)
    echo "ERROR: Unknown precision '${PRECISION}'. Use fp32, fp16, or all."
    exit 1
    ;;
esac

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  Export Complete!"
echo "============================================================"
echo "  Checkpoint:  ${CKPT_DST}"
echo "  ONNX:        ${ONNX_FILE}"
if [ "${PRECISION}" = "fp32" ] || [ "${PRECISION}" = "all" ]; then
  echo "  Engine FP32: ${ENGINE_FP32}"
fi
if [ "${PRECISION}" = "fp16" ] || [ "${PRECISION}" = "all" ]; then
  echo "  Engine FP16: ${ENGINE_FP16}"
fi
echo ""
echo "  To run inference (inside ${TRT_CONTAINER} container):"
echo "    cd /workspace/AV-Solutions/uniad-trt/inference_app/enqueueV3"
echo "    ln -sf /workspace/AV-Solutions/uniad-trt/UniAD/data ./data"
echo "    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH ./build/uniad \\"
echo "      /workspace/AV-Solutions/uniad-trt/<engine_path> \\"
echo "      ./build/libuniad_plugin.so \\"
echo "      /workspace/AV-Solutions/uniad-trt/UniAD/nuscenes_np/uniad_trt_input \\"
echo "      ./output \\"
echo "      69"
echo "============================================================"
