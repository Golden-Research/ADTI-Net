#!/usr/bin/env bash
# ------------------------------------------------------------------------
# ADTI-Net (base model, ResNet-101) evaluation on ImageNet VID val.
# Inference uses N=30 frames per sequence (1 current frame + 29 uniformly
# sampled reference frames), i.e. sequence-wise parallel detection.
# ------------------------------------------------------------------------

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/adti_net/r101_adti_base
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}

python -u main.py \
    --model_type adti \
    --backbone resnet101 \
    --dataset_file vid_multi \
    --num_feature_levels 1 \
    --dilation \
    --num_queries 72 \
    --nheads 4 \
    --num_s_dtd_layers 2 \
    --num_t_dtd_layers 3 \
    --attn_window 256 \
    --max_seq_frames 30 \
    --film_dim 128 \
    --film_tau 0.5 \
    --batch_size 1 \
    --num_ref_frames 4 \
    --num_ref_frames_eval 29 \
    --eval \
    --resume ${EXP_DIR}/checkpoint.pth \
    --num_workers 8 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval.$T
