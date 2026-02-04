
docker run -it --gpus all     --ipc=host     --ulimit memlock=-1     --shm-size=32g\
     -v /localhome/local-samehm/DL4AGX/AV-Solutions/uniad-trt:/workspace/uniad-trt     uniad_torch1.12 /bin/bash


nohup bash -c 'PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/uniad-trt/UniAD_train/UniAD:$PYTHONPATH \
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=28599 \
    ./tools/train.py ./projects/configs/stage1_track_map/full_track_map.py \
    --launcher pytorch --deterministic \
    --work-dir ./projects/work_dirs/stage1_track_map/full_track_map/ \
    --cfg-options data.workers_per_gpu=4 log_config.interval=10 dist_params.backend=gloo' \
    > train_stage1.log 2>&1 &  


