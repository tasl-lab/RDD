#!/bin/bash
set -e -x

echo 'Before setup, follow https://github.com/sled-group/RACER#install-racer step 4 to install CoppeliaSim'
read -p "Have you installed CoppeliaSim? (y/n): " coppeliasim_installed
if [ "$coppeliasim_installed" != "y" ]; then echo "Please install CoppeliaSim first."; exit 1; fi
read -p "Have you installed cuda 11.7? (y/n): " cuda_installed
if [ "$cuda_installed" != "y" ]; then echo "Please install cuda 11.7 first."; exit 1; fi

create_directory_if_not_exists() { [ -d "$1" ] || mkdir -p "$1"; }
absolute_path() {
	local path=$1
	if [ -d "$path" ]; then echo "$(cd "$path" && pwd)";
	elif [ -f "$path" ]; then echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")";
	else echo "Error: $path is not a valid file or directory."; exit 1; fi
}

eval "$(conda shell.bash hook)"
root_path=$(pwd)
switch_cuda_script=$root_path/scripts/setup/switch_cuda.sh
lib_path=3rdparty
model_path=3rdparty/models

create_directory_if_not_exists $lib_path;   lib_path=$(cd $lib_path && pwd);   cd $root_path
create_directory_if_not_exists $model_path; model_path=$(cd $model_path && pwd); cd $root_path

# switch to cuda 11.7 (RACER requirement)
source $switch_cuda_script 11.7
set -e  # switch_cuda.sh runs 'set +e'; re-enable fail-fast for the rest of setup

echo 'Setting up racer environment...'
cd $lib_path
conda create --name racer python=3.8.18 -y
conda activate racer
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install iopath==0.1.9 -c iopath -y
conda install nvidiacub==1.10.0 -c bottler -y
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.5+pt1.13.1cu117
git clone -b main https://github.com/WaterHyacinthInNANHU/RACER
cd RACER
git submodule update --init
pip install -e .
pip install -e libs/PyRep
pip install -e libs/RLbench
pip install -e libs/YARR
pip install -e libs/peract_colab

echo 'Setting up racer_datagen environment...'
cd $lib_path
conda create --name racer_datagen python=3.9 -y
conda activate racer_datagen
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install iopath==0.1.9 -c iopath -y
conda install nvidiacub==1.10.0 -c bottler -y
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.5+pt1.13.1cu116
git clone -b release https://github.com/WaterHyacinthInNANHU/RACER-DataGen
cd RACER-DataGen
git submodule update --init
pip install pip==19.2
pip install -e .
pip install -e racer_datagen/libs/PyRep
pip install -e racer_datagen/libs/RLBench
pip install -e racer_datagen/libs/YARR
pip install -e racer_datagen/libs/peract_colab/
pip install zmq cbor2

echo 'Setting up llava-next environment...'
cd $lib_path
git clone -b racer_llava https://github.com/WaterHyacinthInNANHU/Open-LLaVA-NeXT
cd Open-LLaVA-NeXT
conda create -n llava-next python=3.10 -y
conda activate llava-next
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn==2.7.3 --no-build-isolation
pip install -U accelerate==0.32.0
pip install peft==0.10.0

echo 'Getting models...'
conda activate base
pip install -U "huggingface_hub[cli]"
cd $model_path
huggingface-cli download Yinpei/racer-visuomotor-policy-rich --local-dir racer-visuomotor-policy-rich
cd $root_path
create_directory_if_not_exists 3rdparty/RACER/racer/runs
ln -s $(absolute_path 3rdparty/models/racer-visuomotor-policy-rich) 3rdparty/RACER/racer/runs/racer-visuomotor-policy-rich
cd $model_path
huggingface-cli download lmms-lab/llama3-llava-next-8b --local-dir llama3-llava-next-8b
huggingface-cli download google-t5/t5-11b --local-dir t5-11b
conda activate llava-next
pip install git+https://github.com/openai/CLIP.git

echo 'Done. Activate envs with: conda activate {racer|racer_datagen|llava-next}'
