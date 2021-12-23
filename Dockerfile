FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as nvidia_cuda10

# ubuntu setup
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget git build-essential dialog apt-utils libglib2.0 \
 libsm6 libfontconfig1 libxrender1 libxext6 libgl1-mesa-glx && apt-get clean && rm -rf /var/lib/apt/lists/* && \
 useradd -ms /bin/bash pv

# conda environment setup
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /home/pv/conda && \
    rm ~/miniconda.sh && /home/pv/conda/bin/conda clean -tipsy && \
    /bin/bash -c ". /home/pv/conda/etc/profile.d/conda.sh && conda update -y -n base conda && \
    conda create -y -n pv python=3.6 && conda activate pv && \
    pip install --upgrade pip && conda install -y -c conda-forge uwsgi && \
    pip install matplotlib opencv-python plyfile 'trimesh>=2.35.39,<2.35.40' 'networkx>=2.2,<2.3' scipy tensorboardX &&\
    pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html"

# pointnet++ (requires a CUDA version available in PATH) \
RUN git clone --progress https://github.com/Pointivo/3detr /setup-3detr
WORKDIR /setup-3detr
#ARG TORCH_CUDA_ARCH_LIST="6.1+PTX"
#ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"
ARG TORCH_CUDA_ARCH_LIST="7.0+PTX"
RUN /bin/bash -c ". /home/pv/conda/etc/profile.d/conda.sh && conda activate pv && \
    cd /setup-3detr/third_party/pointnet2 && python setup.py install && \
    cd /setup-3detr/utils && conda install cython -y && python cython_compile.py build_ext --inplace"
