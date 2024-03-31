#!/bin/bash
# Script to set up Isaac Sim on a cluster.
# This script was adapted from here:
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html

set -e

# Create a directory for installation artifacts.
build_dir=$(pwd)build/sim/
mkdir -p ${build_dir}

showmsg() {
  echo -e "\e[1;32m$1\e[0m"
}

showmsg "Updating and installing dependencies..."
sudo apt-get update
sudo apt install build-essential -y

# Install NVIDIA drivers. This shouldn't be necessary since the cluster
# already has NVIDIA drivers installed.
# if [[ ! -f ${build_dir}/NVIDIA-Linux-x86_64-525.85.05.run ]]; then
#   showmsg "Downloading NVIDIA drivers..."
#   wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.85.05/NVIDIA-Linux-x86_64-525.85.05.run -P ${build_dir}
#   chmod +x ${build_dir}/NVIDIA-Linux-x86_64-525.85.05.run
#   sudo ${build_dir}/NVIDIA-Linux-x86_64-525.85.05.run
# fi

# Docker installation using the convenience script
if [[ ! -f /usr/bin/docker ]]; then
  showmsg "Installing Docker..."
  curl -fsSL https://get.docker.com -o ${build_dir}get-docker.sh
  sudo sh ${build_dir}get-docker.sh

  showmsg "Post-install steps for Docker..."
  sudo groupadd docker
  sudo usermod -aG docker $USER
  newgrp docker

  showmsg "Verifying Docker installation..."
  docker run hello-world
fi

# Configure the repository
if [[ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]]; then
  showmsg "Configuring the NVIDIA Container Toolkit repository..."
  if [[ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  fi
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  showmsg "Updating the package list..."
  sudo apt-get update

  showmsg "Installing NVIDIA Container Toolkit..."
  sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
fi

showmsg "Verifying NVIDIA Container Toolkit..."
docker run --rm --gpus all ubuntu nvidia-smi

showmsg "Installing NVIDIA Container Runtime..."
docker login nvcr.io
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1
