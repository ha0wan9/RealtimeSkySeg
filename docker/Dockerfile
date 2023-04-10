# Use the official nvidia CUDA and Cudnn docker image as the base
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Install necessary packages and tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV MINICONDA_VERSION py39_4.10.3
RUN curl -o ~/miniconda.sh -LO https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Set environment variables
ENV PATH /opt/conda/bin:$PATH

# Create conda environment
RUN conda create -n skyseg python=3.9 -y
RUN echo "conda activate skyseg" >> ~/.bashrc

# Install PyTorch and MMSegmentation
RUN /bin/bash -c "source activate skyseg && \
    conda install cudatoolkit=11.3 pytorch=*=*cuda* -c pytorch -y && \
    pip install openmim && \
    mim install mmengine && \
    mim install \"mmcv>=2.0.0\" && \
    pip install git+https://github.com/open-mmlab/mmsegmentation.git"

# Clone the project code 
WORKDIR /
RUN git clone https://github.com/ha0wan9/RealtimeSkySeg.git /RealtimeSkySeg

# Set the working directory to the project root and install the dependencies
WORKDIR /RealtimeSkySeg
RUN pip install -r requirements.txt

# Download the pretrained bisenetv2 model
RUN mim download mmsegmentation --config bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024 --dest models

# Launch a bash shell by default
CMD ["/bin/bash"]