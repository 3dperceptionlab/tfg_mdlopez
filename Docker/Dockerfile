# Description: Dockerfile for the huggingface/transformers-pytorch-gpu image with additional packages
FROM huggingface/transformers-pytorch-gpu:latest

# Install additional packages
RUN apt-get update && apt-get install -y \
    git wget curl libgl1-mesa-glx python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt