#!/bin/bash
set -e -x

create_directory_if_not_exists() {
	local dir_path=$1
	if [ ! -d $dir_path ]; then
		mkdir -p $dir_path
	fi
}
eval "$(conda shell.bash hook)"

env_name=rdd
root_path=$(pwd)
lib_path=3rdparty
model_lib_path=$lib_path/models

# make dirs
create_directory_if_not_exists $lib_path
lib_path=$(cd $lib_path && pwd)
cd $root_path
create_directory_if_not_exists $model_lib_path
model_lib_path=$(cd $model_lib_path && pwd)
cd $root_path

echo "lib_path: $lib_path"
echo "model_lib_path: $model_lib_path"

source activate base && conda activate $env_name

# uvd code base
cd $lib_path
git clone https://github.com/zcczhang/UVD
cd $root_path
# patch UVD requirements.txt
cp .patches/UVD/uvd_requirements.txt  $lib_path/UVD/requirements.txt
cd $lib_path
cd UVD && pip install -e .

# LIV&CLIP
cd $model_lib_path
git clone https://github.com/penn-pal-lab/LIV.git
cd LIV && pip install -e . && cd liv/models/clip && pip install -e .

# set up annoy: https://github.com/spotify/annoy
pip install --user annoy

# other requirements
cd $root_path
pip install -r scripts/setup/requirements.txt

# dirty fix, re-install LIV&CLIP in case of failed dependencies
cd $model_lib_path
cd LIV && pip install -e . && cd liv/models/clip && pip install -e .
python -c "from liv import load_liv; liv = load_liv()"

##### Optional Encoders #####

# # VIP
# cd $model_lib_path
# git clone https://github.com/facebookresearch/vip.git
# cd vip && pip install -e .

# # R3M
# cd $model_lib_path
# git clone https://github.com/facebookresearch/r3m.git
# cd r3m && pip install -e .

# # VC1
# cd $model_lib_path
# git clone https://github.com/facebookresearch/eai-vc.git 
# cd eai-vc && pip install -e vc_models