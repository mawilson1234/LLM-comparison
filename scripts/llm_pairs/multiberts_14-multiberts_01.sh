#!/bin/bash

#SBATCH --job-name=llm-comparison
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load CUDA
module load cuDNN
module load miniconda

source activate llm-divergence

python llm_comparison.py \
	p_model=multiberts_14 \
	q_model=multiberts_01 \
	batch_size=32 \
	kl_masking=always \
	saved_indices=saved_amask_indices.json \
	device=gpu