#!/usr/bin/env bash
MSCOCO_DIR="${HOME}/dataset/coco"
MODEL_DIR="${HOME}/dataset/show_tell_attend/model"

export CUDA_VISIBLE_DEVICES="gpu:3"

bazel-bin/show_tell_attend/evaluate \
    --input_file_pattern="${MSCOCO_DIR}/val-?????-of-00004" \
    --checkpoint_dir="${MODEL_DIR}/train" \
    --eval_dir="${MODEL_DIR}/eval"
