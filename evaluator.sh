sudo docker build -t torch torch_models/ && \
sudo docker run \
    -it --rm --gpus 1 --ipc=host \
    -v /data/sign_language_videos/popsign_v2/563_normalized_hands:/data \
    -v ./meta:/meta \
    -v ./models:/models \
    -v ./detailed_reports:/reports \
    torch python3 detailed_evaluation.py $1