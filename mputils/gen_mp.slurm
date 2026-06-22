#!/bin/bash

declare -a pt_vids

# for testing only
declare -a pts=(3f8b 13e3 494d b2d1 c0df d3ab 8e3b fe96 8c4d a3d4 3a6e 3d12 f9ea 2ff7 e0f7 ed8e 51f5 a362 a6ed 0ba8 812c 03ad a021 a442 1d72 711d a95b fa10 1bd5 6b92 5b63 bd21 1f91 917d fbb7 4ddc ab12 dbf9 99cb 39e5 4f1e 63a1 163a c82a f418 9d2b b718 39a6 4c3d 675f 9b23 9ed9 d478 f066 e3c0 fede 0a77 0bea d05c 9ff4 f760 7f32 80fe 19d3 6f68 a3e7 cf84 d69c 1f86 2f35 e4fa 5d33)

TOTAL_SLURM_TASKS=50

video_dir=/storage/home/hcoda1/5/rsridhar37/scratch/fingerspelling_videos/dmk_v1/video_clips/dmk_v1-train
scripts_dir=/ISLR-ML/mputils

pt_vids=()
threads=`nproc --all`

for pt in ${pts[@]};
do
    while IFS= read -r -d $'\0' file; do
        pt_vids+=("$file")
    done < <(find $video_dir -type f -name "$pt*.mp4" -print0)
done

# Add denominator - 1 to numerator to get ceiling instead of floor
chunk_size=$(((${#pt_vids[@]}+(TOTAL_SLURM_TASKS-1)) / $TOTAL_SLURM_TASKS))
st_idx=$(($SLURM_ARRAY_TASK_ID * $chunk_size))
en_idx=$(($st_idx + $chunk_size))

echo ""
echo "####################"
echo "Num PT Vids: ${#pt_vids[@]}"
echo "Threads: $threads"
echo "Thread No: $SLURM_ARRAY_TASK_ID"
echo "Chunk Size: $chunk_size"
echo "Start: $st_idx End: $en_idx"
echo "####################"
echo ""

echo ""
echo "####################"
echo "Confidence Levels"
echo "Detection: $MIN_DETECTION_CONFIDENCE"
echo "Tracking: $MIN_TRACKING_CONFIDENCE"
echo "####################"
echo ""

for vid in ${pt_vids[@]:$st_idx:$chunk_size};
do
    base_vid=`basename $vid`
    vid_name="${base_vid%.*}"

    out_path="$scripts_dir/data/h5_det${MIN_DETECTION_CONFIDENCE}_trk${MIN_TRACKING_CONFIDENCE}"
    out_h5="$out_path/$vid_name.h5"
    if [ ! -d $out_path ]; then
        mkdir -p $out_path
    fi

    python3 $scripts_dir/main.py hands $vid $out_h5
    echo "Processed: $out_h5. Exit: $?"
done

