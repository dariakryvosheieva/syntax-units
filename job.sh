#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=mit_normal_gpu
python ablation.py --model-name openai-community/gpt2-xl --dataset blimp --ablation-type mean --savedir english/mean-ablation --percentage 1.0
