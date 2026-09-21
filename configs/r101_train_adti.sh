#!/usr/bin/env bash
# ------------------------------------------------------------------------
# ADTI-Net (base model, ResNet-101) on ImageNet VID + DET.
# Implementation details (Sec. IV-C of the paper):
#   * input: 1 current frame + 4 randomly sampled support frames
#     (i.e. --num_ref_frames 4) per training sample;
#   * 72 object queries per frame, 4 attention heads;
#   * 2 spatially-decoupled + 3 temporally-decoupled decoder layers per
#     ADTD branch; local window G = 256;
#   * TFIL: textual/imitation embedding dim 128, tau = 0.5;
#   * shorter side of the input resized to 600 (max 1000), as in the
#     dataloader (datasets/vid_multi.py);
#   * AdamW, lr 1e-4, dropped to 1e-5 at 3/4 of the schedule (paper:
#     120K/160K iterations with total batch size 4).
#
# The paper trains for 160K iterations with a total batch size of 4
# (1 clip per GPU on 4 GPUs). With the ~1.1M training samples of the
# joint set this corresponds to roughly 4 epochs; adjust --epochs /
# --lr_drop_epochs to your iteration budget.
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
    --film_loss_coef 1.0 \
    --batch_size 1 \
    --num_ref_frames 4 \
    --epochs 4 \
    --lr_drop_epochs 3 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --num_workers 8 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
