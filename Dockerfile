from stereolabs/zed:3.7-gl-devel-cuda11.4-ubuntu20.04

arg DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    usbutils \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
	bash /tmp/miniconda.sh -b -p /opt/conda && \
	rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate" >> ~/.bashrc
run conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
run conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   
RUN conda create -y -n visual python=3.10
SHELL ["/opt/conda/bin/conda", "run", "-n", "visual", "bash", "-c"]

run pip install tqdm imageio tyro matplotlib tyro jax
run pip install "wandb<0.18"
run pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
