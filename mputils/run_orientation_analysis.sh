#!/bin/bash

. ./visualization/utils.sh

ORIGINAL_LOGS=${VISUALIZATION_ROOT}/logs/original

for dataset in ${DATASETS[@]}; do
    echo ./analysis.py --action_type sample_videos_by_orientation --video_loc ${FSBOARD_MAP[$dataset]} --sample_size 0 --save_video_loc ${ORIGINAL_VIDEOS}/${dataset} --log_file ${ORIGINAL_LOGS}/${dataset}/orientation_counts.log
done

