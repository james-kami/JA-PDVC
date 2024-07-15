Please obtain your path by running ``pwd``.
The path of my directory is ``/home/james/JA-PDVC``.

1. Clone latest version of James' repo
```
git clone git@github.com:james-kami/JA-PDVC.git
```

2. Pull required files from submodule (.gitmodules)
```
git submodule update --init --recursive
```

4. Create CONDA enviorment. Following steps mainly from CN team.
```
conda create -n james-pdvc python=3.7
```
```
conda activate james-pdvc
```
```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```
```
conda install ffmpeg
```
```
pip install -r requirement.txt
```

5. Compile deformable attention layer
```
cd
cd JA-PDVC/pdvc/ops
sh make.sh
```

6. Download video features from this [link](https://drive.google.com/file/d/19scB8hHNQeLlo0bqYdl7LtkEGKwPSyJq/view?usp=sharing) and then put in c3d folder.
 ``` 
cd JA-PDVC/data/anet/features
mkdir c3d
cd JA-PDVC/data/anet/features/c3d
tar --strip-components=1 -xvf c3d.tar && rm c3d.tar
```


Training
 Start Training with PDVC learnt proposals - dense video captioning
```
cd JA-PDVC/
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path cfgs/anet_c3d_pdvc.yml --gpu_id 0 
```

Evaluation
Start Training with PDVC learnt proposals - dense video captioning
```
eval_folder=/home/james/JA-PDVC/save/anet_c3d_pdvc
python eval.py --eval_folder /home/james/JA-PDVC/save/anet_c3d_pdvc --eval_transformer_input_type queries --gpu_id 0
```

Start feature extraction
Download 3 models (model-best.pth, model-best-anet.pth, and r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth), add custom.yml. add videos.
```
sh run_get_video_feature.sh /home/james/mp4video home/james/JA-PDVC/data/custom /home/james/JA-PDVC/model-best-anet.pth en
```

bash test_and_visualize.sh /home/james/JA-PDVC/visualization/two_vids /home/james/JA-PDVC/visualization/output /home/james/JA-PDVC/save/anet_tsp_pdvc en

python train_custom.py --cfg_path /home/james/JA-PDVC/cfgs/custom.yml --gpu_id 0


> Written with [StackEdit](https://stackedit.io/).
