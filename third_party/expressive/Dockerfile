# Use the official Ubuntu 20.04 LTS image
# docker build -t fbx_image .
# docker run -it fbx_image
FROM ubuntu:20.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME="/home/user"
ENV FBX_SDK_VERSION="2020.3.2"
ENV SIP_VERSION="4.19.3"
ENV PYTHON_VERSION="3.8"
ENV FBX_SDK_PATH="$HOME/fbx_setup/fbx_sdk"
ENV FBX_PYTHON_BINDING_PATH="$HOME/fbx_setup/fbx_python_bindings"
ENV SIP_ROOT="$HOME/fbx_setup/sip"
ENV FBXSDK_ROOT="$FBX_SDK_PATH"
ENV SIP_PATH="$SIP_ROOT/sip-$SIP_VERSION"
# pfb30 fix this
ENV PYTHON_EXECUTABLE="/home/user/miniconda/envs/humanoid/bin/python"

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        zlib1g-dev \
        wget \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        libssl-dev \
        vim \
        libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p $FBX_SDK_PATH && \
    mkdir -p $FBX_PYTHON_BINDING_PATH && \
    mkdir -p $SIP_ROOT

# Install Conda
RUN wget -qO- https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
bash miniconda.sh -b -p $HOME/miniconda && \
# rm miniconda.sh \
$HOME/miniconda/bin/conda init bash

FROM base AS install_fbx

# Make sure the base environment is activated by default
SHELL ["/bin/bash", "-c"]

# Install packages inside the Conda environment
RUN /bin/bash -c "source ~/.bashrc && \
conda create -n humanoid python=3.8 -y && \
conda activate humanoid && \
conda install -y pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch && \
conda install -y 'numpy<1.24' && \
conda install -c conda-forge gxx_linux-64=12.1.0 -y && \
pip install pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown pandas matplotlib ruamel.yaml xlrd scipy"
# conda install -c conda-forge gxx_linux-64=12.1.0 -y && \
FROM install_fbx AS base_conda

# Install FBX SDK
WORKDIR $FBX_SDK_PATH

# Copy the FBX SDK tar.gz into the image
# Place fbx202032_fbxsdk_linux.tar.gz in the same directory as this Dockerfile
COPY fbx202032_fbxsdk_linux.tar.gz $FBX_SDK_PATH/
RUN tar xzf fbx202032_fbxsdk_linux.tar.gz 
RUN chmod +x fbx202032_fbxsdk_linux 
COPY install_fbx.sh .
RUN chmod +x install_fbx.sh
RUN ./install_fbx.sh

FROM base_conda AS fbx_python_bindings
# Install FBX Python Bindings
WORKDIR $FBX_PYTHON_BINDING_PATH

# Copy the FBX Python Bindings tar.gz into the image
# Place fbx202032_fbxpythonbindings_linux.tar.gz in the same directory as this Dockerfile
COPY fbx202032_fbxpythonbindings_linux.tar.gz $FBX_PYTHON_BINDING_PATH/

RUN tar xzf fbx202032_fbxpythonbindings_linux.tar.gz
RUN chmod +x fbx202032_fbxpythonbindings_linux
COPY install_fbx_bindings.sh .
RUN chmod +x install_fbx_bindings.sh
RUN ./install_fbx_bindings.sh

FROM fbx_python_bindings AS sip

# Install SIP
WORKDIR $SIP_ROOT
RUN wget "https://sourceforge.net/projects/pyqt/files/sip/sip-${SIP_VERSION}/sip-${SIP_VERSION}.tar.gz/download" -O "sip-${SIP_VERSION}.tar.gz" && \
    tar xzf "sip-${SIP_VERSION}.tar.gz" && \
    cd sip-${SIP_VERSION} && \
    $PYTHON_EXECUTABLE configure.py && \
    make && \
    make install

ENV SIP_ROOT="$SIP_ROOT/sip-$SIP_VERSION"

FROM sip AS build_fbx

# Build the SDK
WORKDIR $FBX_PYTHON_BINDING_PATH

RUN $PYTHON_EXECUTABLE PythonBindings.py "Python3_x64" buildsip

# Install git
RUN apt-get update && apt-get install -y git

# RUN git clone https://github.com/Ke-Wang1017/expressive_humanoid.git
RUN mkdir -p $HOME/expressive_humanoid
COPY repo_copy/ $HOME/expressive_humanoid/

FROM build_fbx AS rebuild_fbx

WORKDIR $HOME/expressive_humanoid/

# RUN cp CMU_fbx.zip ASE/ase/poselib/data/
# # # # Clone the repository
# # # RUN git clone https://github.com/Ke-Wang1017/expressive_humanoid.git
# # # # sentence-transformers

RUN apt-get update && apt-get install -y unzip

# Modify the Makefile to fix linking order
WORKDIR $FBX_PYTHON_BINDING_PATH/build/Python38_x64

RUN make clean && \
    sed -i 's|\(LIBS = -L[^ ]*\) -lz -lxml2 \([^ ]*\)|\1 \2 -lz -lxml2|' Makefile
# Build and install
RUN make install

# Build the SDK
WORKDIR $FBX_PYTHON_BINDING_PATH
RUN $PYTHON_EXECUTABLE PythonBindings.py "Python3_x64" buildsip


# # RUN PYTHON_SITE_PACKAGES=$($PYTHON_EXECUTABLE -c "import site; print(site.getsitepackages()[0])") && \
# #     mkdir -p $PYTHON_SITE_PACKAGES/fbx && \
# #     cp $FBX_PYTHON_BINDING_PATH/build/Distrib/site-packages/fbx/* $PYTHON_SITE_PACKAGES/fbx/

FROM rebuild_fbx AS final

COPY test_fbx.py .

RUN $PYTHON_EXECUTABLE test_fbx.py
# Set the default command to bash
COPY parse_cmu_mocap_all_2.py $HOME/expressive_humanoid/ASE/ase/poselib/
COPY fbx_importer_all_2.py $HOME/expressive_humanoid/ASE/ase/poselib/
# RUN python parse_cmu_mocap_all_2.py
# python fbx_importer_all_2.py
# RUN /bin/bash -c "source ~/.bashrc && conda activate humanoid && pip install scipy"
# Copy fbx modules to site-packages

CMD ["/bin/bash"]
