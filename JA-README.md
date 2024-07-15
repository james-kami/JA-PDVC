Please obtain your path by running 'pwd'.
The path of my directory is '/home/james/JA-PDVC'.

1. Clone latest version of James' repo
git clone git@github.com:james-kami/JA-PDVC.git

2. Pull required files from submodule (.gitmodules)
git submodule update --init --recursive

3. Create CONDA enviorment. Following steps mainly from CN team.
conda create -n james-pdvc python=3.7

conda activate james-pdvc

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

conda install ffmpeg

pip install -r requirement.txt


4. Compile deformable attention layer
cd
cd JA-PDVC/pdvc/ops
sh make.sh

5. Download video features from this link and then put in c3d folder. 
cd JA-PDVC/data/anet/features
mkdir c3d
cd JA-PDVC/data/anet/features/c3d
tar --strip-components=1 -xvf c3d.tar && rm c3d.tar



1. Start Training with PDVC learnt proposals - dense video captioning
cd JA-PDVC/
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path cfgs/anet_c3d_pdvc.yml --gpu_id 0 
