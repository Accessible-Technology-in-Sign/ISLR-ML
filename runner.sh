#!/bin/bash
input_dir="/data/sign_language_videos/first_round_datasets/250_signs/clean/blue"
for file in "$input_dir"/*.mp4; do
    base_name=$(basename "$file" .mp4)    
    sudo docker compose run hands "$base_name.mp4" "$base_name.h5" #&> /dev/null
done