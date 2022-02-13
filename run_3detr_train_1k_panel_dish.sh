#!/bin/bash
# Full telecom tower dataset (1k)
# Model: 3DETR-m
# in build_preencoder: using nsample=128, 64
# 128 max objs, 24 angle cls bins

nqueries=512
preenc_npoints=5120
dataset="telecomtower"
enc_type="masked"

dataset_dir=/mnt/4tbssd/talha/datasets/telecom-tower-1k/processed_subsampled_2class;
desc="pv_$dataset-1k-$nqueries-q_preenc-$preenc_npoints-$enc_type"
ckpt_dir=/mnt/4tbssd/talha/3detr/ckpts/$(date +"%Y-%m-%d_%H.%M.%s")_train_"$desc";

mkdir $ckpt_dir;
python main.py \
  --nqueries $nqueries \
  --nsemcls 2 \
  --preenc_npoints $preenc_npoints \
  --enc_type $enc_type \
  --enc_nlayers 3 \
  --enc_dim 256 \
  --enc_ffn_dim 128 \
  --enc_nhead 8 \
  --dec_nlayers 8 \
  --dec_dim 256 \
  --dec_ffn_dim 256 \
  --dec_nhead 8 \
  --dataset_name $dataset \
  --dataset_root_dir $dataset_dir \
  --dataset_num_workers 2 \
  --batchsize_per_gpu 1 \
  --max_epoch 2000 \
  --eval_every_epoch 25 \
  --save_separate_checkpoint_every_epoch 50 \
  --log_metrics_every 1 \
  --ngpus 2 \
  --seed 123 \
  --checkpoint_dir $ckpt_dir \
  |& tee -a "$ckpt_dir/logs.txt" ;

echo "Training session saved at: $ckpt_dir";