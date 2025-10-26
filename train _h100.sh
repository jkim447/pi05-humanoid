#!/bin/bash
#SBATCH --job-name=pi05
#SBATCH --output=pi05-%j.out
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris-hgx-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=700G                  # adjust if you need less/more
#SBATCH --cpus-per-task=32          # tune if you want more data-loader threads
#SBATCH --mail-user=jwbkim@stanford.edu
#SBATCH --mail-type=ALL

set -e

# go to your project
cd /iris/projects/humanoid/openpi

# load your env
source .venv/bin/activate
source set_env.sh 

# run training
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_galaxea_egodex_joints \
    --exp-name=galaxea_egodex_joints \
    --overwrite