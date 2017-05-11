#!/usr/bin/env bash
# Directory containing preproceed MSCOCO data
MSCOCO_DIR="${HOME}/dataset/coco"

# VGG_19 checkpoint file
VGG_CHECKPOINT="${HOME}/dataset/show_tell_attend/data/vgg_19.ckpt"

# Directory to save the model
MODEL_DIR="${HOME}/dataset/show_tell_attend/model"

export CUDA_VISIBLE_DEVICES="gpu:1"
# Build the model
bazel build -c opt show_tell_attend/...

# Run the training script
bazel-bin/show_tell_attend/train \
    --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
    --vgg_checkpoint_file="${VGG_CHECKPOINT}" \
    --train_dir="${MODEL_DIR}/train" \
    --train_vgg=false \
    --number_of_steps=1000000

