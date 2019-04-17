#!/bin/bash
export NVVL_DIR=/home/klecki/work/nvvl
export DATA_DIR=/mnt/nvvl_data/data

export NVC="NVIDIA_DRIVER_CAPABILITIES=video,compute,utility"

##nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $NVVL_DIR:/workspace -v $DATA_DIR:/data_dir  -u $(id -u):$(id -g) -p 3567:3567 -p 6006:6006 vsrnet /workspace/examples/pytorch_superres/run_distributed.sh
##nvidia-docker run --rm -it --ipc=host -e $NVC -v $NVVL_DIR:/workspace -v $DATA_DIR:/data_dir  -u $(id -u):$(id -g) -p 3567:3567 -p 6006:6006 vsrnet /workspace/examples/pytorch_superres/run_distributed.sh

#The one that I use:
#nvidia-docker run --rm -it --ipc=host --net=host -e $NVC -v $NVVL_DIR:/workspace -v $DATA_DIR:/data_dir  -u $(id -u):$(id -g) -p 3567:3567 -p 6006:6006 vsrnet
