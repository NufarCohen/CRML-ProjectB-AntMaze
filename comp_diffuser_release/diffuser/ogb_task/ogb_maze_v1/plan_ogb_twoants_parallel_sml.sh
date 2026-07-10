#!/bin/bash

#SBATCH --job-name=script-ev-hi
#SBATCH --output=trash/slurm/plan_OG_StglSml_Jan27/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

### brainiac, 2080: alexa,alexa
#SBATCH --exclude="clippy,voltron,claptrap,alexa,bmo,olivaw,oppy"

##SBATCH --partition="rl2-lab"

#SBATCH --gres=gpu:a40:1
##SBATCH --gres=gpu:l40s:1
##SBATCH --qos="short"

##SBATCH --gres=gpu:rtx_6000:1
##SBATCH --gres=gpu:a5000:1

#SBATCH --qos="debug"
#SBATCH --time=16:00:00

echo "$(hostname)"

source ~/.bashrc
source activate compdfu_ogb_release

# ------------------------------------------------------------------
# Planning: OGBench AntSoccer Arena Stitch Planner -- two-ant CONCURRENT
# (parallel) relay. See plan_ogb_twoants_sml.sh for the sequential variant.
# ------------------------------------------------------------------

config="config/ogb_ant_soc/og_antSoc_Ar_o17d_DiTd768_PadBuf_Ft64_ts512_fs4_h160_ovlp56MditD384.py"


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

GPU_ID=${1:-0}
PLAN_N_EP=${2:-1}
PL_SEEDS=${3:--1}
# 4th arg: 1 = burn subtitles into video (default), 0 = off
VID_SUBTITLES=${4:-1}

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${GPU_ID} \
MUJOCO_EGL_DEVICE_ID=${GPU_ID} \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_twoants_parallel_sml.py \
    --config "${config}" \
    --plan_n_ep "${PLAN_N_EP}" \
    --pl_seeds "${PL_SEEDS}" \
    --is_vid_subtitles "${VID_SUBTITLES}"

exit 0
