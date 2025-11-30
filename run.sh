#!/bin/bash
DATASET_NAME="Tmars"

#CUDA_VISIBLE_DEVICES=0 \
#python train.py \
#--img_aug \
#--batch_size 32 \
#--dataset_name $DATASET_NAME \
#--loss_names 'msc+vtc' \
#--num_epoch 60 \
#--MLM \
#--eval_period 5 \
#--sampler 'random' \
#--test_mode 'rss' \
#--pretrain_choice 'ViT-B/32'
#--root_dir '/media/jqzhu/e/jjsu/datasets'
#RN50 RN101 ViT-B/32 RN50x4 RN50x16 ViT-L/14
CUDA_VISIBLE_DEVICES=0 \
python test.py
