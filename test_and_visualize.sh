#!/bin/bash -i
curdir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP/data
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP/extract_features
export PYTHONPATH=$PYTHONPATH:$curdir/visualization

DATA_PATH=$1
OUTPUT_FOLDER=$2
PDVC_MODEL_PATH=$3
OUTPUT_LANGUAGE=$4

if [ -z "$DATA_PATH" ] || [ -z "$OUTPUT_FOLDER" ] || [ -z "$PDVC_MODEL_PATH" ]; then
    echo "Please ensure all arguments (DATA_PATH, OUTPUT_FOLDER, PDVC_MODEL_PATH, OUTPUT_LANGUAGE) are provided."
    exit 1
fi

RELEASED_CHECKPOINT="r2plus1d_34-tsp_on_activitynet" # Set this to the correct model checkpoint
METADATA_CSV_FILENAME="$DATA_PATH/metadata.csv"
FEATURE_DIR="$OUTPUT_FOLDER/${RELEASED_CHECKPOINT}_stride_16/"
STRIDE=16
SHARD_ID=0
NUM_SHARDS=1
DEVICE=cuda
WORKER_NUM=8

#echo "START GENERATE METADATA"
#python video_backbone/TSP/data/generate_metadata_csv.py --video-folder $DATA_PATH --output-csv $METADATA_CSV_FILENAME

mkdir -p $FEATURE_DIR

#echo "START EXTRACT VIDEO FEATURES"
#python video_backbone/TSP/extract_features/extract_features.py \
#--data-path $DATA_PATH \
#--metadata-csv-filename $METADATA_CSV_FILENAME \
#--released-checkpoint $RELEASED_CHECKPOINT \
#--stride $STRIDE \
#--shard-id $SHARD_ID \
#--num-shards $NUM_SHARDS \
#--device $DEVICE \
#--output-dir $FEATURE_DIR \
#--workers $WORKER_NUM

echo "START Dense-Captioning"
python eval.py --eval_mode test \
--eval_save_dir $OUTPUT_FOLDER \
--eval_folder generated_captions \
--eval_model_path $PDVC_MODEL_PATH \
--test_video_feature_folder $FEATURE_DIR \
--test_video_meta_data_csv_path $METADATA_CSV_FILENAME \

echo "START VISUALIZATION"
python visualization/visualization.py \
--input_mp4_folder $DATA_PATH \
--output_mp4_folder $OUTPUT_FOLDER/vis_videos \
--dvc_file $OUTPUT_FOLDER/generated_captions/dvc_results.json \
--output_language $OUTPUT_LANGUAGE

