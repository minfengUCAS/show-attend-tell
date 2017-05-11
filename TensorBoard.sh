#!/usr/bin/env bash
MODEL_DIR="${HOME}/dataset/show_tell_attend/model"

tensorboard --logdir="${MODEL_DIR}" --port=8008
