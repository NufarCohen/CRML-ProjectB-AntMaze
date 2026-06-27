#!/bin/bash
#SBATCH --job-name=ogb
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

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python smoke_test.py

end_time=$(date +%s)
echo "Training ended at: $(date)"

# Calculate the difference
elapsed=$((end_time - start_time))
echo "Total Elapsed Time: $(($elapsed / 3600))h $(($elapsed % 3600 / 60))m $(($elapsed % 60))s"
exit 0





