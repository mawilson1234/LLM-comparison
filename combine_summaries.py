import re
import os
import pandas as pd

from glob import glob

def combine_summaries() -> None:
	'''Combines all summary CSVs in outputs.'''
	fs = [f for f in glob('outputs/**', recursive=True) if f.endswith('summary.csv.gz')]
	combined = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
	
	for c in ['p_model', 'q_model']:
		combined[c] = [m.replace('google/', '') for m in combined[c]]
		combined[c] = [re.sub('-seed_([0-9]{1})(-|$)', '-seed_0\\1\\2', m) for m in combined[c]]
		combined[c] = [re.sub('-seed', '', m) for m in combined[c]]
	
	combined.q_model = [re.sub('-step_([0-9]{1})k', '-step_000\\1k', m) for m in combined.q_model]
	combined.q_model = [re.sub('-step_([0-9]{2})k', '-step_00\\1k', m) for m in combined.q_model]
	combined.q_model = [re.sub('-step_([0-9]{3})k', '-step_0\\1k', m) for m in combined.q_model]
	combined.q_model = [m.replace('-step', '').replace('k', '') for m in combined.q_model]
	
	combined = combined.sort_values(['p_model', 'q_model'], kind='stable')
	
	combined.to_csv(os.path.join('outputs', 'summaries.csv.gz'), index=False)	

if __name__ == '__main__':

	combine_summaries()