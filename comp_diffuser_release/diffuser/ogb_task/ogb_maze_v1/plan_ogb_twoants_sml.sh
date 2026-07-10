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
# Config
# ------------------------------------------------------------------

# config=""

# config="config/og_antM_Gi_o2d_luotest.py"  ## test only, placeholder config file


# ------------------------------------------------------------------
# Planning: OGBench AntMaze Stitch 15/29D Planner
# i.e., the diffusion planner generates trajectory of shape (Horizon, 15/29)
# ------------------------------------------------------------------

# config="config/ogb_ant_maze/og_antM_Lg_o15d_PadBuf_Ft64_ts512_DiTd768dp16_fs4_h160_ovlp56Mdit.py"
# config="config/ogb_ant_maze/og_antM_Lg_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"


# ------------------------------------------------------------------
# Planning: OGBench AntMaze Stitch 2D Planner
# ------------------------------------------------------------------

# Giant
# config="config/ogb_ant_maze/og_antM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"

# Large
# config="config/ogb_ant_maze/og_antM_Lg_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"

# Medium
# config="config/ogb_ant_maze/og_antM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"


# ------------------------------------------------------------------
# Planning: OGBench AntMaze Explore 2D Planner
# ------------------------------------------------------------------

# TODO:


# ------------------------------------------------------------------
# Planning: OGBench PointMaze Stitch 2D Planner
# ------------------------------------------------------------------

# Giant
# config="config/ogb_pnt_maze/og_pntM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"

# Large
# config="config/ogb_pnt_maze/og_pntM_Lg_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"

# Medium
# config="config/ogb_pnt_maze/og_pntM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"


# ------------------------------------------------------------------
# Planning: OGBench AntSoccer Arena Stitch Planner
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
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_twoants_sml.py \
    --config "${config}" \
    --plan_n_ep "${PLAN_N_EP}" \
    --pl_seeds "${PL_SEEDS}" \
    --is_vid_subtitles "${VID_SUBTITLES}"

exit 0