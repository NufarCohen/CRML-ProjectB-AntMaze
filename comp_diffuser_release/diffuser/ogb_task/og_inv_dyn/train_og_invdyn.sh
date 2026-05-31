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

source ~/.bashrc
source /home/projects/crml-prj10844/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release

cd /home/projects/crml-prj10844/comp_diffuser_release

# 1. הגדרות סביבה גרפית לשרת ללא מסך (Headless EGL)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYOPENGL_FORCE_HEADLESS=1
unset DISPLAY

# 2. התיקון ההיסטורי - טעינה מראש של הספרייה שאישרת שקיימת בשרת
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1

# 3. נתיבי ספריות ומודולים
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
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

# קונפיגורציית הריצה שלך
config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g2d_invdyn_h12.py"

echo "USING CONFIG=$config"
grep -n "dataset =" "$config" || true
grep -n "antmaze-" "$config" || true

# הרצת הסקריפט בצורה נקייה ומקורית
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
python diffuser/ogb_task/og_inv_dyn/train_og_invdyn.py --config $config

end_time=$(date +%s)
echo "Training ended at: $(date)"

# חישוב זמן הריצה של ה-Job
elapsed=$((end_time - start_time))
echo "Total Elapsed Time: $(($elapsed / 3600))h $(($elapsed % 3600 / 60))m $(($elapsed % 60))s"

exit 0