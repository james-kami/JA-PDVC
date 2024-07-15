#!/bin/bash -i
curdir=`pwd`
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP/data
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP/extract_features
export PYTHONPATH=$PYTHONPATH:$curdir/visualization

DATA_PATH=$1 # path of the raw video folder
FEATURE_DIR=$2 # path of the output folder to save generated captions
PDVC_MODEL_PATH=$3
OUTPUT_LANGUAGE=$4

if [ -z "$DATA_PATH" ]; then
    echo "DATA_PATH variable is not set."
    echo "Please set DATA_PATH to the folder containing the videos you want to process."
    exit 1
fi

if [ -z "$FEATURE_DIR" ]; then
    echo "OUTPUT_FOLDER variable is not set."
      echo "Please set OUTPUT_FOLDER to the folder you want to save generate captions."
    exit 1
    exit 1
fi

if [ -z "$PDVC_MODEL_PATH" ]; then
    echo "PDVC_MODEL_PATH variable is not set."
    echo "Please set the pretrained PDVC model path (only support PDVC with TSP features)."
    exit 1
fi

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

METADATA_CSV_FILENAME=$DATA_PATH/"metadata.csv" # 之后会生成关于视频的信息csv的地址【不用改】
RELEASED_CHECKPOINT=r2plus1d_34-tsp_on_activitynet
#mkdir -p $OUTPUT_DIR
# Choose the stride between clips, e.g. 16 for non-overlapping clips and 1 for dense overlapping clips
STRIDE=16

# Optional: Split the videos into multiple shards for parallel feature extraction
# Increase the number of shards and run this script independently on separate GPU devices,
# each with a different SHARD_ID from 0 to NUM_SHARDS-1.
# Each shard will process (num_videos / NUM_SHARDS) videos.
SHARD_ID=0
NUM_SHARDS=1
DEVICE=cuda
WORKER_NUM=8

echo "START GENERATE METADATA"

CUDA_VISIBLE_DEVICES=1 python video_backbone/TSP/data/generate_metadata_csv.py --video-folder $DATA_PATH --output-csv $METADATA_CSV_FILENAME

# FEATURE_DIR=$OUTPUT_FOLDER/${RELEASED_CHECKPOINT}_stride_${STRIDE}/

echo "START EXTRACT VIDEO FEATURES"
CUDA_VISIBLE_DEVICES=1 python video_backbone/TSP/extract_features/extract_features.py \
--data-path $DATA_PATH \
--metadata-csv-filename $METADATA_CSV_FILENAME \
--released-checkpoint $RELEASED_CHECKPOINT \
--stride $STRIDE \
--shard-id $SHARD_ID \
--num-shards $NUM_SHARDS \
--device $DEVICE \
--output-dir $FEATURE_DIR \
--workers $WORKER_NUM \
--local-checkpoint '/home/james/JA-PDVC/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth'
echo "End EXTRACT VIDEO FEATURES"
#cn数据集全部转为MP4存在 /mnt/mp4video
#参数：mp4位置  特征输出位置  ckpt位置  中英文选择  
#bash run_get_video_feature.sh /mnt/mp4video /home/lanyun/xjtu/cm/video-mamba-suite/video-mamba-suite/video-dense-captioning/data/custom /home/lanyun/xjtu/cm/video-mamba-suite/video-mamba-suite/video-dense-captioning/pretrain/model-best.pth en
#bash run_get_video_feature.sh /mnt/mp4video /home/lanyun/xjtu/lyz/PDVC-lyz/data/custom /home/lanyun/xjtu/lyz/PDVC-lyz/model-best-anet.pth en
# sh run_get_video_feature.sh /home/james/mp4video home/james/JA-PDVC/data/custom /home/james/JA-PDVC/model-best-anet.pth en
