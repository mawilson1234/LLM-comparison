import os
import re
import sys
from itertools import permutations

def generate_scripts(d: str = '.') -> None:
	'''
	Generates scripts for running KL divergence comparisons.
	Currently pairs each model with every other model, except that intermediate MultiBERT checkpoints
	are not used as baseline models, and intermediate MultiBERT checkpoints are only compared to their
	corresponding final state.
	
		params:
			d (str): a directory to save the scripts in (optional, default is current directory)
	'''
	if not os.path.isdir(d):
		os.mkdir(d)
	
	configs = sorted([f.replace('.yaml', '') for f in os.listdir('models')])
	pairs 	= permutations(configs,2)
	
	# we don't want to use the intermediate multiberts as baseline models, 
	# as they're not good stand-ins for the true distribution
	pairs 	= [pair for pair in pairs if not pair[0].endswith('k')]
	
	# we only want to compare the intermediate multiberts to their corresponding final state as a baseline
	pairs	= [pair for pair in pairs if not pair[1].endswith('k') or re.sub(r'-(.*)k$', '', pair[1]) == pair[0]]
	
	# we don't need to use the redundant final checkpoint of the multibert models, since they're the same as the baseline version
	pairs 	= [pair for pair in pairs if not pair[1].endswith('2000k')]
	
	header 	= '\n'.join((
		'#!/bin/bash',
		'',
		'#SBATCH --job-name=kl-comparison',
		'#SBATCH --output=joblogs/%x_%j.txt',
		'#SBATCH --nodes=1',
		'#SBATCH --cpus-per-task=1',
		'#SBATCH --mem=16G',
		'#SBATCH --time=48:00:00',
		'#SBATCH --gpus=v100:1',
		'#SBATCH --partition=gpu',
		'#SBATCH --mail-type=END,FAIL,INVALID_DEPEND',
		'',
		'module load CUDA',
		'module load cuDNN',
		'module load miniconda',
		'',
		'source activate kl-divergence',
		'',
	))
	
	for p_model, q_model in pairs:
		file = header + (
			f'\npython kl_comparison.py p_model={p_model} q_model={q_model}'
		)
		
		with open(os.path.join(d, f'{p_model}-{q_model}.sh'), 'wt') as out_file:
			out_file.write(file)

if __name__ == '__main__':
	generate_scripts(
		os.path.join('scripts', sys.argv[-1]) 
		if not sys.argv[-1].endswith('create-script-pairs.py') 
		else os.path.join('scripts', '.')
	)
