#!/bin/bash
#SBATCH --job-name=pi05
#SBATCH --output=pi05-%j.out
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --qos=normal
#SBATCH --nodelist=iris-hgx-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --mem=800G                  # adjust if you need less/more
#SBATCH --cpus-per-task=24          # tune if you want more data-loader threads

set -e

# go to your project
cd /iris/projects/humanoid/openpi

# load your env
source .venv/bin/activate
source set_env.sh 

# run training
srun --ntasks=1 --gpus-per-task=2 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_mixed \
    --exp-name=co_training_50_demos_per_task_6gpus \
    --overwrite