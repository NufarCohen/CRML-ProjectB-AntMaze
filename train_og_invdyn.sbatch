#!/bin/bash
#SBATCH --job-name=ogb_diffusion_with_val
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --partition=debug
#SBATCH --qos=normal
echo "Running on host: $(hostname)"
source ~/.bashrc
source /home/projects/crml-prj10844/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release

cd /home/projects/crml-prj10844/comp_diffuser_release

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "===== ENV CHECK ====="
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "DISPLAY=$DISPLAY"
echo "CONDA_PREFIX=$CONDA_PREFIX"

python - <<'EOF'
import os
print("PY MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PY PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))
print("PY DISPLAY =", os.environ.get("DISPLAY"))

import mujoco.gl_context
print("GLContext =", mujoco.gl_context.GLContext)
EOF

echo "Training started at: $(date)"
start_time=$(date +%s)

## optionally 'cd' to 'comp_diffuser_release' folder
# cd $Folder

# echo $(hostname)


## ------------ AntMaze Explore ------------
## AntMaze Medium Explore
# config="config/maze_og/og_inv/og_antMexpl_Me_o29d_g2d_invdyn_h12.py"

## AntMaze Large Explore
# config="config/maze_og/og_inv_Jan1/og_antMexpl_Lg_o29d_g2d_invdyn_h12.py"

## -----------------------------------------


## humanoid
# config="config/maze_og/og_inv_Jan4/og_humM_Gi_o69d_g2d_invdyn_h80_dm5_dout02.py"
# config="config/maze_og/og_inv_Jan4/og_humM_Lg_o69d_g2d_invdyn_h80_dm5_dout02.py"
# config="config/maze_og/og_inv_Jan4/og_humM_Me_o69d_g2d_invdyn_h80_dm5_dout02.py"


## ------------ AntSoccer Arena ------------
# config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Ar_o42d_g4d_invdyn_h120_dout02.py"
config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Ar_o42d_g17d_invdyn_h120_dout02.py"

## ------------ AntSoccer Medium ------------
# config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Me_o42d_g4d_invdyn_h120_dm4_dout02.py"
# config="config/ogb_invdyn/og_inv_ant_soc/og_antSoc_Me_o42d_g17d_invdyn_h100_dm4_dout02.py"



## --------- Training Inverse Dynamics Models: OGBench AntMaze Stitch 2D Planner ----------

## ** Example: **
## the state space of ant is 29D (so observation is 29D)
## 2D (x-y) diffusion planner (so goal is 2D)
## to set which inverse dynamics model to use, check 'diffuser/utils/ogb_utils/ogb_serial.py'

## ant maze giant stitch
#config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g2d_invdyn_h12.py" ## for 2D planner
# config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g15d_invdyn_h12.py"
# config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g29d_invdyn_h12_dm5.py"

## Large
## ant maze large stitch
# config="config/ogb_invdyn/og_inv_ant/og_antM_Lg_o29d_g2d_invdyn_h12.py"
#config="config/ogb_invdyn/og_inv_ant/og_antM_Lg_o29d_g15d_invdyn_h12.py"
#config="config/ogb_invdyn/og_inv_ant/og_antM_Lg_o29d_g29d_invdyn_h12.py"

## Medium
## ant maze medium stitch
#config="config/ogb_invdyn/og_inv_ant/og_antM_Me_o29d_g2d_invdyn_h12.py"
# config="cconfig/ogb_invdyn/og_inv_ant/og_antM_Me_o29d_g15d_invdyn_h12.py"
#config="config/ogb_invdyn/og_inv_ant/og_antM_Me_o29d_g29d_invdyn_h12_dm5.py"


PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/ogb_task/og_inv_dyn/train_og_invdyn.py --config $config

end_time=$(date +%s)
echo "Training ended at: $(date)"

# Calculate the difference
elapsed=$((end_time - start_time))
echo "Total Elapsed Time: $(($elapsed / 3600))h $(($elapsed % 3600 / 60))m $(($elapsed % 60))s"
exit 0





