#!/bin/sh
train="python turngpt/train.py --batch_size 1024 --load_from_checkpoint --pretrained_model_name_or_path runs/TurnGPT/TurnGPT_kf09yt99/epoch=0_val_loss=15.3017.ckpt --gpus -1  --trp_projection_steps 1 --resume must --id kf09yt99"

$train
