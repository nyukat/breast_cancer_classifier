#!/bin/bash

DEVICE_TYPE='gpu'
NUM_EPOCHS=10
HEATMAP_BATCH_SIZE=100
GPU_NUMBER=0

PATCH_MODEL_PATH='models/sample_patch_model.p'
IMAGE_MODEL_PATH='models/ImageOnly__ModeImage_weights.p'
IMAGEHEATMAPS_MODEL_PATH='models/ImageHeatmaps__ModeImage_weights.p'

DENSENET_MODEL_TF_MAP_PATH='models/densenet_weights_map.json'
RESNET_MODEL_TF_MAP_PATH='models/resnet_weights_map.json'

SAMPLE_SINGLE_OUTPUT_PATH='sample_single_output'
export PYTHONPATH=$(pwd):$PYTHONPATH


echo 'Stage 1: Crop Mammograms'
python3 src/cropping/crop_single.py \
    --mammogram-path $1 \
    --view $2 \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl

echo 'Stage 2: Extract Centers'
python3 src/optimal_centers/get_optimal_center_single.py \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl

echo 'Stage 3: Generate Heatmaps'
python3 src/heatmaps/run_producer_single_tf.py \
    --model-path ${PATCH_MODEL_PATH} \
    --tf-torch-weights-map-path ${DENSENET_MODEL_TF_MAP_PATH} \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl \
    --batch-size ${HEATMAP_BATCH_SIZE} \
    --heatmap-path-malignant ${SAMPLE_SINGLE_OUTPUT_PATH}/malignant_heatmap.hdf5 \
    --heatmap-path-benign ${SAMPLE_SINGLE_OUTPUT_PATH}/benign_heatmap.hdf5\
    --device-type ${DEVICE_TYPE} \
    --gpu-number ${GPU_NUMBER}

echo 'Stage 4a: Run Classifier (Image)'
python3 src/modeling/run_model_single_tf.py \
    --view $2 \
    --model-path ${IMAGE_MODEL_PATH} \
    --tf-torch-weights-map-path ${RESNET_MODEL_TF_MAP_PATH} \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl \
    --use-augmentation \
    --num-epochs ${NUM_EPOCHS} \
    --device-type ${DEVICE_TYPE} \
    --gpu-number ${GPU_NUMBER}

echo 'Stage 4b: Run Classifier (Image+Heatmaps)'
python3 src/modeling/run_model_single_tf.py \
    --view $2 \
    --model-path ${IMAGEHEATMAPS_MODEL_PATH} \
    --tf-torch-weights-map-path ${RESNET_MODEL_TF_MAP_PATH} \
    --cropped-mammogram-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped.png \
    --metadata-path ${SAMPLE_SINGLE_OUTPUT_PATH}/cropped_metadata.pkl \
    --use-heatmaps \
    --heatmap-path-malignant ${SAMPLE_SINGLE_OUTPUT_PATH}/malignant_heatmap.hdf5 \
    --heatmap-path-benign ${SAMPLE_SINGLE_OUTPUT_PATH}/benign_heatmap.hdf5\
    --use-augmentation \
    --num-epochs ${NUM_EPOCHS} \
    --device-type ${DEVICE_TYPE} \
    --gpu-number ${GPU_NUMBER}
