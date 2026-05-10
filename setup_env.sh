#!/bin/bash
set -e 

echo "======================================================="
echo "  Starting MinkowskiEngine (CUDA 12.1 / PyTorch 2.4.0) "
echo "======================================================="

echo -e "\n>>> Updating system and installing GCC 12 & Build Essentials..."
sudo apt-get update
sudo apt-get install -y wget git build-essential cmake libopenblas-dev gcc-12 g++-12

if [ ! -d "$HOME/miniconda3" ]; then
    echo -e "\n>>> Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
else
    echo -e "\n>>> Miniconda is already installed. Skipping..."
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
echo -e "\n>>> Accept conda ToS"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
echo -e "\n>>> Installing Mamba to the base environment..."
conda install -n base -c conda-forge mamba -y

if [ ! -d "/usr/local/cuda-12.1" ]; then
    echo -e "\n>>> Downloading and installing System CUDA 12.1 Development Toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run -O cuda_12.1.run
    sudo sh cuda_12.1.run --silent --toolkit --override
    rm cuda_12.1.run
else
    echo -e "\n>>> CUDA 12.1 toolkit is already installed in /usr/local/cuda-12.1. Skipping..."
fi

echo -e "\n>>> Creating Mamba environment 'intobj' with Python 3.9..."
mamba create -n intobj python=3.9 -y
conda activate intobj

echo -e "\n>>> Installing PyTorch 2.4.0 and dependencies via Mamba..."
mamba install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -y openblas-devel -c anaconda
mamba install -y nvidia/label/cuda-12.1.0::cuda-toolkit

echo -e "\n>>> Cloning custom MinkowskiEngine branch (cuda12-installation)..."
if [ -d "MinkowskiEngine" ]; then
    echo "Directory MinkowskiEngine already exists. Removing old version..."
    rm -rf MinkowskiEngine
fi
git clone https://github.com/CiSong10/MinkowskiEngine.git
cd MinkowskiEngine
git checkout cuda12-installation

echo -e "\n>>> Setting environment variables for the RTX 3090..."
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CC=gcc-12
export CXX=g++-12

export TORCH_CUDA_ARCH_LIST="8.6"

echo -e "\n>>> Compiling MinkowskiEngine (MAX_JOBS=4)..."
MAX_JOBS=4 python setup.py install --blas=openblas --force_cuda

echo -e "\n>>> Installing remaining project dependencies via pip..."
pip install open3d trimesh numpy tensorboard pyviz3d h5py

echo "======================================================="
echo "  Setup Complete! MinkowskiEngine is ready to run.     "
echo "  To use your environment, type: conda activate intobj "
echo "======================================================="
