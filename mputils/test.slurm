#!/bin/bash

declare -a pt_vids
declare -a pts=("63a1" "19d3")
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

echo "####################"
echo "Num PT Vids: ${#pt_vids[@]}"
echo "Threads: $threads"
echo "Thread No: $SLURM_ARRAY_TASK_ID"
echo "Chunk Size: $chunk_size"
echo "Start: $st_idx End: $en_idx"
echo "####################"
echo ""

for vid in ${pt_vids[@]:$st_idx:$chunk_size};
do
    base_vid=`basename $vid`
    vid_name="${base_vid%.*}"

    out_path="$scripts_dir/data"
    out_h5="$out_path/$vid_name.h5"
    python3 $scripts_dir/main.py hands $vid $out_h5
    echo "Processed: $out_h5. Exit: $?"
done

