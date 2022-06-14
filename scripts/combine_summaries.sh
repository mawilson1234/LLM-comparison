#!/bin/bash

#SBATCH --job-name=combine_llm-comparison
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate llm-comparison

python combine_summaries.py amask
