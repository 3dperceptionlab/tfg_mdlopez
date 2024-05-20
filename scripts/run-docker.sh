#!/bin/bash

export containerName=tfg_$USER

docker run -d --gpus "device=0" --rm -it \
    --volume="/home/mdlopez/workspace/:/workspace:rw" \
	--volume="/mnt/md1/datasets/:/datasets:ro" \
    --workdir="/workspace" \
	--name $containerName \
	--shm-size=16g \
	mdlopez/tfg bash
