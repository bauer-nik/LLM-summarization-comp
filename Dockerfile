# linux based python 10 - deepseek needs python 10
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to bypass any interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    curl \
    ca-certificates \
    gnupg2 \
    lsb-release \
    nano \
    && rm -rf /var/lib/apt/lists/*

# move over docker specific requirements
COPY docker_requirements.txt .
# torch install separate for CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# install rest of required packages
RUN pip3 install --no-cache-dir -r docker_requirements.txt

# move app into working directory
WORKDIR /app
COPY app.py /app/

CMD ["/bin/bash"]