import re
import os
import sys
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from glob import glob

def combine_summaries(d: str, in_filename: str, out_filename: str) -> None:
	'''Combines all CSVs with filename in_filename in outputs and output as out_filename.'''
	fs = [f for f in glob(os.path.join('outputs', d, '**'), recursive=True) if os.path.split(f)[-1] == in_filename]
	combined = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
	
	for c in ['p_model', 'q_model']:
		combined[c] = [m.replace('google/', '') for m in combined[c]]
		combined[c] = [re.sub('-seed_([0-9]{1})(-|$)', '-seed_0\\1\\2', m) for m in combined[c]]
		combined[c] = [re.sub('-seed', '', m) for m in combined[c]]
		combined[c] = [re.sub('-step_([0-9]{1})k', '-step_000\\1k', m) for m in combined[c]]
		combined[c] = [re.sub('-step_([0-9]{2})k', '-step_00\\1k', m) for m in combined[c]]
		combined[c] = [re.sub('-step_([0-9]{3})k', '-step_0\\1k', m) for m in combined[c]]
		combined[c] = [m.replace('-step', '').replace('k', '') for m in combined[c]]
	
	combined = combined.sort_values(['p_model', 'q_model'], kind='stable')
	
	combined.to_csv(os.path.join('outputs', d, out_filename), index=False)	

def plot_summaries(d: str, f: str) -> None:
	'''Plots results from summary file.'''
	summary = pd.read_csv(os.path.join(d, f))
	
	with PdfPages(os.path.join('outputs', d, 'plots.pdf')) as pdf:
		for name, expr, size, fmt in [('mean_kl_div', '_0[0-9]_', 3, '.2f'), ('mean_kl_div_zeros', '_0000', 8, '.4f')]:
			if name == 'mean_kl_div':
				kl_div_summary = summary[
						(~summary.p_model.str.contains(expr, regex=True)) & 
						(~summary.q_model.str.contains(expr, regex=True))
					]
			else:
				kl_div_summary = summary[
					(summary.p_model.str.contains(expr)) & 
					(summary.q_model.str.contains(expr))
				]
			
			kl_div_summary = kl_div_summary[
					['p_model', 'q_model', 'mean_kl_div']
				] \
				.pivot(index='q_model', columns='p_model', values='mean_kl_div')
			
			kl_div_summary = kl_div_summary.fillna(0)
			
			ax = sns.heatmap(
				kl_div_summary,
				xticklabels=1,
				yticklabels=1,
				annot=True,
				annot_kws=dict(
					size=size
				),
				fmt=fmt
			)
			
			ax.set_xlabel(ax.get_xlabel().replace('_', ' '))
			ax.set_ylabel(ax.get_ylabel().replace('_', ' '))
			
			fig = plt.gcf()
			fig.suptitle('Mean KL divergence')
			pdf.savefig(bbox_inches='tight')
			plt.close('all')
			del fig
		
		for model in ['multiberts_00', 'multiberts_01', 'multiberts_02', 'multiberts_03', 'multiberts_04']:
			kl_div_summary = summary[
					(summary.p_model.str.contains(model)) &
					(summary.q_model.str.contains(model))
				][
					['p_model', 'q_model', 'mean_kl_div', 'sem_kl_div']
				]
			
			ax = sns.barplot(
				data=kl_div_summary,
				x='q_model',
				y='mean_kl_div',
				hue='p_model',
			)
			
			ax.set_xticklabels(
				ax.get_xticklabels(), rotation=90
			)
			
			plt.legend([], [], frameon=False)
			
			for _, row in kl_div_summary.iterrows():
				ax.errorbar(
					x=row['q_model'], 
					y=row['mean_kl_div'], 
					yerr=row['sem_kl_div'],
					color='black',
					ls='none'
				)
			
			ax.set_xlabel('checkpoint')
			ax.set_ylabel(ax.get_ylabel().replace('_', ' '))
			
			fig = plt.gcf()
			fig.suptitle(f'Mean KL divergence from {model}')
			pdf.savefig(bbox_inches='tight')
			plt.close('all')
			del fig
		
		for metric in ['mean_p_entropy', 'mean_p_logprob']:
			sem = metric.replace('mean', 'sem')
			
			metric_summary = summary[
					~summary.p_model.str.contains('_0000')
				][
					['p_model', metric, sem]
				] \
				.drop_duplicates(ignore_index=True)
			
			# this is to set the hue to default and not color the bars by value
			metric_summary['____'] = 'tmp'
			
			ax = sns.barplot(
				data=metric_summary,
				x='p_model',
				y=metric,
				hue='____'
			)
			
			ax.set_xticklabels(
				ax.get_xticklabels(), rotation=90
			)
			
			plt.legend([], [], frameon=False)
			
			for _, row in metric_summary.iterrows():
				ax.errorbar(
					x=row['p_model'], 
					y=row[metric], 
					yerr=row[sem],
					color='black',
					ls='none'
				)
			
			ax.set_xlabel('model')
			ax.set_ylabel(ax.get_ylabel().replace('_p_', ' '))
			
			fig = plt.gcf()
			fig.suptitle(f'{metric.replace("_p_", " ").capitalize()} of target positions')
			pdf.savefig(bbox_inches='tight')
			plt.close('all')
			del fig
		
		for metric in ['mean_q_entropy', 'mean_q_logprob']:
			for model in ['multiberts_00', 'multiberts_01', 'multiberts_02', 'multiberts_03', 'multiberts_04']:
				sem = metric.replace('mean', 'sem')
				
				metric_summary = summary[
						summary.q_model.str.startswith(model)
					][
						['q_model', metric, sem]
					] \
					.drop_duplicates(ignore_index=True) \
					.groupby('q_model') \
					.first() \
					.reset_index()
					
				metric_summary.q_model = [re.sub('_0([0-9])$', '_0\\1_2000', m) for m in metric_summary.q_model]
				metric_summary = metric_summary.sort_values('q_model').reset_index(drop=True)
				
				# this is to set the hue to default and not color the bars by value
				metric_summary['____'] = 'tmp'
				
				ax = sns.barplot(
					data=metric_summary,
					x='q_model',
					y=metric,
					hue='____'
				)
				
				plt.legend([], [], frameon=False)
				
				for _, row in metric_summary.iterrows():
					ax.errorbar(
						x=row['q_model'], 
						y=row[metric], 
						yerr=row[sem],
						color='black',
						ls='none'
					)
				
				xticklabels = ax.get_xticklabels()
				for l in xticklabels:
					l.set_text(l.get_text().replace('_2000', ''))
				
				ax.set_xticklabels(xticklabels, rotation=90)
				
				ax.set_xlabel('checkpoint')
				ax.set_ylabel(ax.get_ylabel().replace('_q_', ' '))
				
				fig = plt.gcf()
				fig.suptitle(f'{metric.replace("_q_", " ").capitalize()} of target positions for {model}')
				pdf.savefig(bbox_inches='tight')
				plt.close('all')
				del fig
	
if __name__ == '__main__':
	if sys.argv[-1] != 'combine_summaries.py':
		d = sys.argv[-1] if sys.argv[-1] != 'combine_summaries.py'
	
	combine_summaries(d, 'summary.csv.gz', 'summaries.csv.gz')
	combine_summaries(d, 'results.csv.gz', 'merged_results.csv.gz')
	plot_summaries(d, 'summaries.csv.gz')