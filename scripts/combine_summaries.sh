#!/bin/bash

#SBATCH --job-name=combine_kl-divergence
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate kl-divergence

python combine_summaries.py
