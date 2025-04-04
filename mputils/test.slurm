#!/bin/bash

declare -a pt_vids

# for testing only
# declare -a pts=("63a1" "19d3")

declare -a pts=("03ad" "0a77" "0ba8" "0bea" "13e3" "163a" "1bd5" "1f86" "1f91" "2f35" "2ff7" "39a6" "39e5" "3a6e" "3d12" "494d" "4c3d" "4ddc" "51f5" "5b63" "5d33" "675f" "6b92" "6f68" "711d" "7f32" "80fe" "812c" "8c4d" "8e3b" "917d" "99cb" "9b23" "9d2b" "9ed9" "9ff4" "a021" "a362" "a3d4" "a3e7" "a442" "a6ed" "a95b" "b2d1" "b718" "bd21" "c82a" "d05c" "d478" "d69c" "dbf9" "e3c0" "e4fa" "ed8e" "f066" "f418" "f9ea" "fbb7" "fe96" "fede")

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

