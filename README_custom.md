
## Preparation
Environment: Linux,  GCC>=5.4, CUDA >= 9.2, Python>=3.7, PyTorch>=1.5.1

1. Clone the repo
```bash
git clone --recursive https://github.com/ttengwang/PDVC.git
```

2. Create virtual environment by conda
```bash
conda create -n PDVC python=3.7
source activate PDVC
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install ffmpeg
pip install -r requirement.txt
```

3. Compile the deformable attention layer (requires GCC >= 5.4). 
```bash
cd pdvc/ops
sh make.sh
```

## Running PDVC on Your Own Videos
Download a pretrained model ([GoogleDrive](https://drive.google.com/drive/folders/1Y34puRNE0lpbz3i38k1nh8d9E1i3S0i4?usp=drive_link)) with [TSP](https://github.com/HumamAlwassel/TSP) features  and put it into `./save`. Then run:
```bash
video_folder=visualization/videos
output_folder=visualization/output
pdvc_model_path=save/anet_tsp_pdvc/model-best-anet.pth
output_language=en
bash test_and_visualize.sh $video_folder $output_folder $pdvc_model_path $output_language
```
check the `$output_folder`, you will see a new video with embedded captions. 
Note that we generate non-English captions by translating the English captions by GoogleTranslate. 
To produce Chinese captions, set `output_language=zh-cn`. 
For other language support, find the abbreviation of your language at this [url](https://github.com/lushan88a/google_trans_new/blob/main/constant.py), and you also may need to download a font supporting your language and put it into `./visualization`.

## Training and Validation

### Download Video Features

```bash
cd data/anet/features
bash download_anet_c3d.sh
# bash download_anet_tsn.sh
# bash download_i3d_vggish_features.sh
# bash download_tsp_features.sh
```
The preprocessed C3D features have been uploaded to [baiduyun drive](https://pan.baidu.com/s/1Ehvq1jNiJrhgA00mOG25zQ?pwd=fk2p)
### Dense Video Captioning
1. PDVC with learnt proposals
```
# Training
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path ${config_path} --gpu_id ${GPU_ID}
# The script will evaluate the model for every epoch. The results and logs are saved in `./save`.

# Evaluation
eval_folder=anet_c3d_pdvc # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type queries --gpu_id ${GPU_ID}
```

## Train on custom data

```bash
# 1.Extract visual features from MP4 files and save them in Feature output location
bash run_get_video_feature.sh /mnt/mp4video /home/lanyun/xjtu/lyz/PDVC-lyz/data/custom /home/lanyun/xjtu/lyz/PDVC-lyz/model-best-anet.pth en
# Parameters meaning: mp4 file loication;  Feature output location;  PDVC ckpt location; caption language selection

# 2. spilt the data into train set and validation set
python split.py
# You can modify the file path in the script according to your needs.

# 3.Start training
python train_custom.py --cfg_path /home/lanyun/xjtu/lyz/PDVC-lyz/cfgs/custom.yml --gpu_id ${GPU_ID}
# The script will evaluate the model for every epoch. The results and logs are saved in `./save`.
```

bypy upload /home/lanyun/xjtu/lyz/PDVC-lyz DVC/code overwrite
bypy upload /mnt/mp4video DVC/video overwrite