#!/bin/bash
#SBATCH --job-name=pi05
#SBATCH --output=pi05-%j.out
#SBATCH --partition=iris
#SBATCH --nodelist=iris-hgx-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
#SBATCH --mem=1000G                  # TODO: add appropriate memory request
#SBATCH --cpus-per-task=60          # TODO: adjust num cpus
#SBATCH --mail-user=jwbkim@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --account=iris

set -e

# go to your project
cd /iris/projects/humanoid/openpi

# load your env
source .venv/bin/activate
source set_env.sh 

# run trainingss
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_galaxea_egodex_abs_joints \
    --exp-name=galaxea_egodex_abs_joints_composition_redo_redo_redo_with_new_robot_data_11102025 \
    --overwrite