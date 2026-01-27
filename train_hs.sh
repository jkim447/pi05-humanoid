#!/bin/bash
#SBATCH --job-name=pi05
#SBATCH --output=pi05-%j.out
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris-hgx-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --mem=350G                  # adjust if you need less/more
#SBATCH --cpus-per-task=40          # tune if you want more data-loader threads
#SBATCH --mail-user=jwbkim@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --account=iris

set -e

# go to your project
cd /iris/projects/humanoid/openpi

# load your env
source .venv/bin/activate
source set_env.sh 

uv run train_hs.py