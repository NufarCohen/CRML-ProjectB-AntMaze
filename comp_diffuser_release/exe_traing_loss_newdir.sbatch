#!/bin/bash
#SBATCH --job-name=ogb_diffusion_with_val_newdir
#SBATCH --output=/home/projects/crml-prj10844/slurm_logs/slurm-%j.out
#SBATCH --error=/home/projects/crml-prj10844/slurm_logs/slurm-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --partition=debug
#SBATCH --qos=normal

source ~/.bashrc
source /home/projects/crml-prj10844/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release

cd /home/projects/crml-prj10844/comp_diffuser_release

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PYTHONPATH=$PYTHONPATH:$(pwd)

RUN_ID=$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-manual}
RUN_PREFIX="diffusion/${RUN_ID}/"
RUN_DIR="logs/antmaze-giant-stitch-v0/${RUN_PREFIX}og_antM_Gi_o2d_luotest_T512"

export WANDB_NAME="$RUN_ID"
export WANDB_RUN_GROUP="antmaze-giant-stitch-v0"

echo "===== ENV CHECK ====="
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "DISPLAY=$DISPLAY"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "RUN_ID=$RUN_ID"
echo "WANDB_NAME=$WANDB_NAME"
echo "WANDB_RUN_GROUP=$WANDB_RUN_GROUP"
echo "New checkpoints/logs directory will be:"
echo "$PWD/$RUN_DIR"

python - <<'PYEOF'
import os
print("PY MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PY PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))
print("PY DISPLAY =", os.environ.get("DISPLAY"))
print("PY WANDB_NAME =", os.environ.get("WANDB_NAME"))
print("PY WANDB_RUN_GROUP =", os.environ.get("WANDB_RUN_GROUP"))

import mujoco.gl_context
print("GLContext =", mujoco.gl_context.GLContext)
PYEOF

echo "Training started at: $(date)"
start_time=$(date +%s)

python diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py \
--config config/og_antM_Gi_o2d_luotest.py \
--prefix "$RUN_PREFIX" \
--val_ratio 0.05 \
--val_freq 5000

end_time=$(date +%s)
echo "Training ended at: $(date)"

elapsed=$((end_time - start_time))
echo "Total Elapsed Time: $(($elapsed / 3600))h $(($elapsed % 3600 / 60))m $(($elapsed % 60))s"