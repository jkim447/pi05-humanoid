#!/bin/bash
#SBATCH --job-name=pi05
#SBATCH --output=pi05-%j.out
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris-hgx-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --mem=1000G                  # adjust if you need less/more
#SBATCH --cpus-per-task=64          # tune if you want more data-loader threads

set -e

# go to your project
cd /iris/projects/humanoid/openpi

# load your env
source .venv/bin/activate
source set_env.sh 

# run training

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_mixed_kp \
    --exp-name=co_training_kp \
    --resume