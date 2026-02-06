#!/bin/bash
#SBATCH --job-name=pi05
#SBATCH --output=pi05-%j.out
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris7
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --mem=200G                  # TODO: add appropriate memory request
#SBATCH --cpus-per-task=20         # TODO: adjust num cpus
#SBATCH --mail-user=jwbkim@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --account=iris

set -e

# go to your project
cd /iris/projects/humanoid/openpi

# load your env
source .venv/bin/activate
source set_env.sh 

# run trainings
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_galaxea_egodex_abs_joints \
    --exp-name=galaxea_egodex_abs_joints_trash_sorting_robot_data_only_baseline_02052026 \
    --overwrite