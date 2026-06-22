#!/bin/bash

. utils.sh

############################## Generate from supplemental gen within nan ranges ##############################
# ./overlay.py -vs ${SL_VIDEOS_ROOT}/dmk_v1/video_clips/dmk_v1-train/ -p ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.1_0.3.parquet -d overlay_videos/nan_rng0.1_0.3/
# ./overlay.py -vs ${SL_VIDEOS_ROOT}/dmk_v1/video_clips/dmk_v1-train/ -p ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.3_0.5.parquet -d overlay_videos/nan_rng0.3_0.5/
# ./overlay.py -vs ${SL_VIDEOS_ROOT}/dmk_v1/video_clips/dmk_v1-train/ -p ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.5_0.7.parquet -d overlay_videos/nan_rng0.5_0.7/
# ./overlay.py -vs ${SL_VIDEOS_ROOT}/dmk_v1/video_clips/dmk_v1-train/ -p ${MPUTILS_ROOT}/out/landmarks/supplemental_gen_nan_rng0.7_0.9.parquet -d overlay_videos/nan_rng0.7_0.9/

############################## Generate random sample of 500 from supplemental_gen/main_train ##############################
# ./overlay.py -ss 500 -vs ${FSBOARD_ROOT}/dmk_v1/video_clips/dmk_v1-train/ -pf ${MPUTILS_ROOT}/out/landmarks/supplemental_gen.parquet -dbg
# ./overlay.py -ss 500 -vs ${FSBOARD_ROOT}/daun_v1/video_clips/daun_v1-train/ -pf ${MPUTILS_ROOT}/out/landmarks/main_train.parquet -dbg

# ############################## Generate random sample of 10 from supplemental_gen orientation classification videos ##############################
OVERLAY_VIDEOS=${VISUALIZATION_ROOT}/videos/overlay

for dataset in ${DATASETS[@]}; do
for suffix in ${ORIENTATION_SUFFIXES[@]}; do
    ./overlay.py \
        -ss 20 \
        -vs ${FSBOARD_MAP[$dataset]} \
        -fv ${ORIGINAL_VIDEOS}/${dataset}/orientation-${suffix} \
        -pf ${MPUTILS_ROOT}/out/landmarks/${dataset}.parquet
done
done

