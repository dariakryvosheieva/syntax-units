#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:a100:1
python lexical_violation_from_blimp.py