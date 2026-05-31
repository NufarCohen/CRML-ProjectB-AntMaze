#!/bin/bash

source ~/.bashrc

source activate compdfu_ogb_release

## config: e.g., eval problems for which env?
sub_conf='ogb_antM_Gi_Navi_ev_prob_numEp20_eSdSt0_preAct5'

## NOTE: set this to the respective path in your computer
h5_root='../comp_diffuser_release/data/ogb_maze/ev_probs/'
# ../../

{

CUDA_VISIBLE_DEVICES=${1:-0} \
python -B data_gen_scripts/gen_probs_luo/gen_maze_probs.py \
    --sub_conf $sub_conf \
    --h5_root $h5_root

exit 0
}