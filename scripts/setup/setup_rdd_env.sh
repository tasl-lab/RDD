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
cd UVD && git checkout e7d657a35912273604edeb1f1d7016bdfc206596
cd $root_path
# patch UVD requirements.txt
cp .patches/UVD/uvd_requirements.txt  $lib_path/UVD/requirements.txt
cd $lib_path
cd UVD && pip install -e .

# LIV&CLIP
cd $model_lib_path
git clone https://github.com/penn-pal-lab/LIV.git
cd LIV && git checkout a12991f53aab01f3cecc1315a81068ba12e2bd6b
pip install -e . && cd liv/models/clip && pip install -e .

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
# cd vip && git checkout f98ca99bcaf9f7e00ccd1c99735d77f55c53dd69
# pip install -e .

# # R3M
# cd $model_lib_path
# git clone https://github.com/facebookresearch/r3m.git
# cd r3m && git checkout b2334e726887fa0206962d7984c69c5fb09cceab
# pip install -e .

# # VC1
# cd $model_lib_path
# git clone https://github.com/facebookresearch/eai-vc.git 
# cd eai-vc && git checkout 76fe35e87b1937168f1ec4b236e863451883eaf3
# pip install -e vc_models