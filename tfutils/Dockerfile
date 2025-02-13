FROM ubuntu:22.04@sha256:0e5e4a57c2499249aafc3b40fcd541e9a456aab7296681a3994d631587203f97
ENV DEBIAN_FRONTEND=nointeractive
WORKDIR /service
RUN apt update && apt install -y python3 python3-pip python3-venv
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY ./ ./
CMD ["python3", "converter.py"]