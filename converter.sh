sudo docker build -t tfutils tfutils/ && \
sudo docker run \
    -it --rm --gpus 1\
    -v ./model_temps:/saves \
    -v ./tflites:/models \
    tfutils python3 converter.py --type $1 --name $2 --frames $3 --features $4 --signs $5 --lstm $6