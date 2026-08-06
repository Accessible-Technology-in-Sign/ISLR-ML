# utils.sh contains common constants and universal functions

VISUALIZATION_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FSBOARD_ROOT=/data/sign_language_videos/fingerspelling_videos

# DATASETS=(main_train supplemental_gen supplemental_gen_tst)
# DATASET_VIDEOS=(${FSBOARD_ROOT}/daun_v1/video_clips/daun_v1-train/ ${FSBOARD_ROOT}/dmk_v1/video_clips/dmk_v1-train/ ${FSBOARD_ROOT}/dmk_v1/video_clips/dmk_v1-train/)
DATASETS=(supplemental_gen)
DATASET_VIDEOS=(${FSBOARD_ROOT}/dmk_v1/video_clips/dmk_v1-train/)

ORIENTATION_SUFFIXES=(cap0 cap270 fshp2688-1512 fshp1080-1920 fshp1920-960 fshp3264-1836 fshp960-1920 metaFalse)
PARTICIPANTS=(ab12 8e3b)

MPUTILS_ROOT=${VISUALIZATION_ROOT}/..
ORIGINAL_VIDEOS=${VISUALIZATION_ROOT}/videos/original
OVERLAY_VIDEOS=${VISUALIZATION_ROOT}/videos/overlay

declare -A FSBOARD_MAP
for i in ${!DATASETS[@]}; do
    FSBOARD_MAP[${DATASETS[i]}]=${DATASET_VIDEOS[i]}
done
# FSBOARD_MAP[${DATASETS[0]}]=${FSBOARD_ROOT}/daun_v1/video_clips/daun_v1-train/
# FSBOARD_MAP[${DATASETS[1]}]=${FSBOARD_ROOT}/dmk_v1/video_clips/dmk_v1-train/

