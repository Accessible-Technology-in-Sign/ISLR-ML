#!/bin/bash

declare -a pt_vids
declare -a pts=("$1")

# video_dir=/storage/home/hcoda1/5/rsridhar37/scratch/fingerspelling_videos/dmk_v1/video_clips/dmk_v1-train
# scripts_dir=/ISLR-ML/mputils
video_dir=/data/sign_language_videos/fingerspelling_videos/dmk_v1/video_clips/dmk_v1-train
scripts_dir=/data/deep_learning/ISLR-ML/mputils

pt_vids=()

for pt in ${pts[@]};
do
    while IFS= read -r -d $'\0' file; do
        pt_vids+=("$file")
    done < <(find $video_dir -type f -name "$pt*.mp4" -print0)
done

# Add denominator - 1 to numerator to get ceiling instead of floor

echo ""
echo "####################"
echo "Num PT Vids: ${#pt_vids[@]}"
echo "####################"
echo ""

for vid in ${pt_vids[@]};
do
    base_vid=`basename $vid`
    vid_name="${base_vid%.*}"

    out_path="$scripts_dir/data/h5_data"
    out_h5="$out_path/$vid_name.h5"
    
    python3 $scripts_dir/main.py hands $vid $out_h5
    echo "Processed: $out_h5. Exit: $?"
done

