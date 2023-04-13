#!/bin/bash

#SBATCH --job-name=backup_llm-comparison
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

backup outputs/
