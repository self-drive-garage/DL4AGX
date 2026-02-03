
## Model Training and Exportation
### Training
Two model variants are provided: UniAD-tiny for edge deployment (Jetson Orin) and UniAD-full for desktop GPUs with more VRAM.

| model | img backbone | bev size | img size | bev encoder layers | with bevslicer? | FPN levels | with bev upsample? |
| :---: | :---: | :---: | :---: | :---: | :---:|:---:| :---: |
| UniAD-full | ResNet-101| 200x200  | 1600x928 | 6 | Y | 4 | N |
| UniAD-tiny | ResNet-50 | 50x50 | 400x256 | 3 | N | 1 | N |


Please follow [training instructions](https://github.com/OpenDriveLab/UniAD/blob/main/docs/TRAIN_EVAL.md) from official UniAD for details on UniAD model training.

---

### UniAD-tiny

To train this variant, the following files are needed:

1. Configs: [stage1](../projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py) and [stage2](../projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py) for `UniAD-tiny` training.

2. Download BEVFormer-tiny weights from [BEVFormer Model Zoo](https://github.com/fundamentalvision/BEVFormer?tab=readme-ov-file#model-zoo) for stage1 initialization.

Launch UniAD-tiny training and evaluation in a separate training docker container and separate UniAD project\
Step 1: Create a separate training project from scratch: clone a UniAD Repo to a separate project and checkout, apply patch to support torch-1.12 training, borrow `third_party.uniad_mmdet3d`, copy tools and configs, and download weights
```
cd uniad-trt
mkdir UniAD_train && cd UniAD_train
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
git checkout 02fa68c5
git apply <path_to_uniad-trt/patch/uniad-tiny-training-support.patch>
cp -r <path_to_uniad-trt/UniAD/third_party> .
cp <path_to_uniad-trt/tools/postprocess_bevformer_tiny_epoch_24_pth.py> ./tools/
cp <path_to_uniad-trt/projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py> ./projects/configs/stage1_track_map/
cp <path_to_uniad-trt/projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py> ./projects/configs/stage2_e2e/
mkdir ckpts
wget -O ./ckpts/bevformer_tiny_epoch_24.pth <link_to_BEVFormer-tiny_weights>
```
Step 2: prepare nuscenes dataset to `<uniad-trt/UniAD_train/UniAD/data>`\
Step 3: launch a training docker container for UniAD-tiny training, compile `third_party.uniad_mmdet3d` and modify BEVformer-tiny pretrained weights
```
docker run -it --gpus all --shm-size=8g -v </host/system/path/to/UniAD_train/UniAD>:/workspace/UniAD_train/UniAD uniad_torch1.12 /bin/bash
cd /workspace/UniAD_train/UniAD/third_party/uniad_mmdet3d/
python3 setup.py build develop --user
cd /workspace/UniAD_train/UniAD
python3 ./tools/postprocess_bevformer_tiny_epoch_24_pth.py
```
Step 4: train and evaluate
```
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py NUM_GPUs
cp ./projects/work_dirs/stage1_track_map/tiny_imgx0.25_track_map/epoch_6.pth ./ckpts/
mv ./ckpts/epoch_6.pth ./ckpts/tiny_imgx0.25_track_map.pth
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py NUM_GPUs
cp ./projects/work_dirs/stage2_e2e/tiny_imgx0.25_e2e/epoch_20.pth ./ckpts/
mv ./ckpts/epoch_20.pth ./ckpts/tiny_imgx0.25_e2e.pth
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py ./ckpts/tiny_imgx0.25_e2e.pth 1
```

---

### UniAD-full

To train the full-size variant, the following files are needed:

1. Configs: [stage1](../projects/configs/stage1_track_map/full_track_map.py) and [stage2](../projects/configs/stage2_e2e/full_e2e.py) for `UniAD-full` training.

2. Download BEVFormer-base (ResNet-101-DCN) weights from [BEVFormer Model Zoo](https://github.com/fundamentalvision/BEVFormer?tab=readme-ov-file#model-zoo) for stage1 initialization. You need the **base** model, not the tiny one.

Launch UniAD-full training and evaluation in a separate training docker container and separate UniAD project\
Step 1: Create a separate training project from scratch: clone a UniAD Repo to a separate project and checkout, apply patch to support torch-1.12 training, borrow `third_party.uniad_mmdet3d`, copy tools and configs, and download weights
```
cd uniad-trt
mkdir UniAD_train && cd UniAD_train
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
git checkout 02fa68c5
git apply <path_to_uniad-trt/patch/uniad-tiny-training-support.patch>
cp -r <path_to_uniad-trt/UniAD/third_party> .
cp <path_to_uniad-trt/projects/configs/stage1_track_map/full_track_map.py> ./projects/configs/stage1_track_map/
cp <path_to_uniad-trt/projects/configs/stage2_e2e/full_e2e.py> ./projects/configs/stage2_e2e/
mkdir ckpts
wget -O ./ckpts/bevformer_r101_dcn_24ep.pth <link_to_BEVFormer-base_weights>
```
Step 2: prepare nuscenes dataset to `<uniad-trt/UniAD_train/UniAD/data>`\
Step 3: launch a training docker container for UniAD-full training, compile `third_party.uniad_mmdet3d`
```
docker run -it --gpus all --shm-size=32g -v </host/system/path/to/UniAD_train/UniAD>:/workspace/UniAD_train/UniAD uniad_torch1.12 /bin/bash
cd /workspace/UniAD_train/UniAD/third_party/uniad_mmdet3d/
python3 setup.py build develop --user
cd /workspace/UniAD_train/UniAD
```
> **Note:** The full-size model uses significantly more memory than the tiny variant (200x200 BEV vs 50x50, full-resolution 1600x928 images). Use `--shm-size=32g` or higher for the docker container. Each GPU should have at least 32GB VRAM. With 48GB A40 GPUs, `samples_per_gpu=1` should work. If you encounter OOM during training, reduce `queue_length` from 5 to 3 in the stage1 config.

Step 4: train and evaluate
```
# Stage 1: track + map (6 epochs)
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/full_track_map.py NUM_GPUs
cp ./projects/work_dirs/stage1_track_map/full_track_map/epoch_6.pth ./ckpts/
mv ./ckpts/epoch_6.pth ./ckpts/full_track_map.pth

# Stage 2: end-to-end (20 epochs)
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/full_e2e.py NUM_GPUs
cp ./projects/work_dirs/stage2_e2e/full_e2e/epoch_20.pth ./ckpts/
mv ./ckpts/epoch_20.pth ./ckpts/full_e2e.pth

# Evaluate
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/full_e2e.py ./ckpts/full_e2e.pth 1
```

For multi-node distributed training (e.g., 4 nodes x 8 A40 GPUs), run on each node:
```
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=4 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=28599 \
  ./tools/train.py \
  ./projects/configs/stage1_track_map/full_track_map.py \
  --launcher pytorch
```

---

#### File Structure

After training, put the final checkpoint into the deployment project `<uniad-trt/UniAD/ckpts>` and make sure the structure of `<uniad-trt/UniAD>` is as follows:
```
UniAD
├── ckpts/
│   ├── tiny_imgx0.25_e2e_ep20.pth   # (if using tiny)
│   ├── full_e2e_ep20.pth             # (if using full)
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
├── nuscenes_np/
│   ├── uniad_onnx_input/
│   ├── uniad_trt_input/
├── projects/
├── third_party/
│   ├── uniad_mmdet3d/
├── tools/
```

### Pytorch to ONNX
To export an ONNX model, inside the deployment docker container, please run the following commands.

For UniAD-tiny:
```
cd /workspace/UniAD
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py ./ckpts/tiny_imgx0.25_e2e_ep20.pth 1
```

For UniAD-full:
```
cd /workspace/UniAD
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh ./projects/configs/stage2_e2e/full_e2e_trt_p.py ./ckpts/full_e2e_ep20.pth 1
```

The export script automatically detects model dimensions (BEV size, image resolution) from the config file. UniAD-tiny produces `uniad_tiny_imgx0.25_cp.onnx` and UniAD-full produces `uniad_full_cp.onnx`.

Due to legal reasons, we can only provide an [ONNX](../onnx/uniad_tiny_dummy.onnx) model of UniAD-tiny with random weights. Please follow instructions on training to obtain a model with real weights.

<- Last Page: [Data Preparation](data_prep.md)

-> Next Page: [Explicit Quantization](explicit_quantization.md)
