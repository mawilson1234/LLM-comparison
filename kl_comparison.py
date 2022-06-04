# Compares all pairs of models defined in ./models/ on a given dataset
import os
import hydra
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import KLDivLoss

from tqdm import tqdm
from typing import *
from omegaconf import OmegaConf, DictConfig
from itertools import permutations

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import logging as lg
lg.set_verbosity_error()

from datasets import load_dataset, Dataset, DatasetDict
from datasets.utils import logging as dataset_utils_logging
from datasets.utils import disable_progress_bar
disable_progress_bar()
dataset_utils_logging.set_verbosity_error()

log = logging.getLogger(__name__)

def apply_to_all_of_type(
	data: 'any', 
	t: Type, 
	fun: Callable, 
	*args: Tuple, 
	**kwargs: Dict
) -> 'any':
	'''
	Apply a function to recursively to all elements in an iterable that match the specified type
		
		params:
			data (any)		: an object to recursively apply a function to
			t (type)		: the type of object within data to apply the function to
			fun (Callable)	: the function to apply to any values within data of type t
			*args (tuple)	: passed to fun
			**kwargs (dict)	: passed to fun
		
		returns:
			data (any)		: the data object with fun applied to everything in data matching type t
	'''
	if isinstance(data,(DictConfig,ListConfig)):
		# we need the primitive versions of these so we can modify them
		data = OmegaConf.to_container(data)
	
	data = deepcopy(data)
	
	if isinstance(data,t):
		returns = filter_none(fun(data, *args, **kwargs))
	elif isinstance(data,dict):
		returns = filter_none({apply_to_all_of_type(k, t, fun, *args, **kwargs): apply_to_all_of_type(v, t, fun, *args, **kwargs) for k, v in data.items()})
	elif isinstance(data,(list,tuple,set)):
		returns = filter_none(type(data)(apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data))
	elif isinstance(data,Dataset):
		# this means you have to cast the result back to a dataset afterward. I can't figure out how to do it inplace
		# since it can only be done from a dict but the dict is one level up
		returns = filter_none(list(apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data))
	elif isinstance(data,(torch.Tensor,pd.Series)):
		returns = filter_none(type(data)([apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data]))
	elif isinstance(data,np.ndarray):
		returns = filter_none(np.array([apply_to_all_of_type(i, t, fun, *args, **kwargs) for i in data]))
	else:
		returns = filter_none(data)
	
	if isinstance(data,(pd.Series,np.ndarray)):
		return returns if returns.any() or returns.size == 1 and returns[0] == 0 else None
	else:
		return returns

def filter_none(data: 'any') -> 'any':
	'''
	Remove None values recursively from iterables (i.e., [[None, None]] -> {nothing}, [[None, 'blah']] -> [['blah']])
	
		params:
			data (any)	: any data from which to remove None values
		
		returns:
			data (any)	: the data with None values removed
	'''
	# from https://stackoverflow.com/questions/20558699/python-how-recursively-remove-none-values-from-a-nested-data-structure-lists-a
	data = deepcopy(data)
	
	if isinstance(data,(list,tuple,set)):
		return type(data)(filter_none(x) for x in data if x is not None)
	elif isinstance(data,dict):
		return type(data)(
			(filter_none(k), filter_none(v))
				for k, v in data.items() if k is not None and v is not None
			)
	else:
		if isinstance(data,(pd.Series,np.ndarray)):
			return data if data.any() or data.size == 1 and data[0] == 0 else None
		else:
			return data

def recursor(
	t: 'type', 
	*args: Tuple, 
	**kwargs: Dict
) -> Callable:
	'''
	Creates recursive functions that apply to a single data type and do not rely on output from a previous step
		
		params:
			t (type)		: the type to apply the function to
			*args (tuple)	: passed to the called function
			**kwargs (dict)	: passed to the called function
	'''
	return lambda fun: \
		lambda data, *args, **kwargs: \
			apply_to_all_of_type(data=data, t=t, fun=fun, *args, **kwargs)

@recursor(str)
def format_data_for_tokenizer(
	data: str, 
	string_id: str
) -> str:
	'''
	Format a string for use with a tokenizer
	Recursor means that this applies recursively to any nested data structure, formatting all tokens,
	and outputs data in the same shape as the input
	
		params:
			data (str)			: the data to format for use with a tokenizer
			mask_token (str)	: the tokenizer's mask token
			string_id (str)		: the huggingface string id for the tokenizer
		
		returns:
			data (str)			: the data formatted for use with the tokenizer in string_id
	'''
	data = data.lower() if 'uncased' in string_id or 'multiberts' in string_id else data
	return data

def load_format_dataset(
	dataset_loc: str,
	split: str,
	data_field: str,
	string_id: str,
	n_examples: int = None,
	tokenizer_kwargs: Dict = {},
) -> Dataset:
	'''
	Loads and formats a huggingface dataset according to the passed options
	
		params:
			dataset_loc (str)		: the location of the dataset
			split (str)				: the split to load from the dataset
			data_field (str)		: the field in the dataset that contains the actual examples
			string_id (str)			: a string_id corresponding to a huggingface pretrained tokenizer
									  used to determine appropriate formatting
			n_examples (int)		: how many (random) examples to draw from the dataset
									  if not set, the full dataset is returned
			fmt (str)				: the file format the dataset is saved in
			tokenizer_kwargs (dict)	: passed to AutoTokenizer.from_pretrained
		
		returns:
			dataset (Dataset)		: a dataset that has been formatted for use with the tokenizer,
									  possible with punctuation stripped
	'''
	dataset 				= DatasetDict.load_from_disk(dataset_loc)[split]
	
	# gather these so we know the position of items prior to shuffling 
	# and we can figure out what to convert to torch.Tensors later
	original_cols 			= list(dataset.features.keys())
	original_pos			= list(range(dataset.num_rows))
	
	# cast to dict since datasets cannot be directly modified
	dataset 				= dataset.to_dict()
	dataset['original_pos']	= original_pos
	original_cols.append('original_pos')
	
	baseline_tokenizer 		= AutoTokenizer.from_pretrained(string_id, **tokenizer_kwargs)
	mask_token 				= baseline_tokenizer.mask_token
	dataset[data_field]		= format_data_for_tokenizer(data=dataset[data_field], string_id=string_id)
	
	# now cast back to a dataset
	dataset 				= Dataset.from_dict(dataset)
	dataset 				= sample_from_dataset(dataset, n_examples)
	dataset 				= dataset.map(lambda ex: baseline_tokenizer(ex[data_field]))
	
	dataset.set_format(type='torch', columns=[f for f in dataset.features if not f in original_cols])
	
	return dataset

def sample_from_dataset(
	dataset: Union[Dataset,Dict], 
	n_examples: int = None, 
	log_message: bool = True
) -> Dataset:
	'''
	Draws n_examples random examples from a dataset.
	Does not shuffle if the number of examples >= the length of the dataset
	
		params:
			dataset (Dataset)	: the dataset to draw examples from
			n_examples (int)	: how many examples to draw
			log_message (bool)	: whether to display a message logging the number of examples drawn
								  compared to the size of the full dataset
		
		returns:
			dataset (Dataset)	: a dataset with n_examples random selections from the passed dataset
	'''
	n_examples	= dataset.num_rows if n_examples is None else min(n_examples, dataset.num_rows)
	
	if n_examples < dataset.num_rows:
		if log_message:
			log.info(f'Drawing {n_examples} random examples from {len(dataset)} total')
		
		dataset = dataset.shuffle().select(range(n_examples))
	
	return dataset

def mask_input(
	inputs: torch.Tensor,
	tokenizer: 'PreTrainedTokenizer',
	indices: List[int] = None, 
	masking_style: str = 'always',
	device: str = 'cpu',
) -> torch.Tensor:
	'''
	Creates a masked input according to the specified masking style.
	If indices are provided, mask them according to the masking style.
	Otherwise, choose a random 15% of the indices to mask.
	
	Then:
		If masking is "none", return the original tensor.
		If masking is "always", replace indices with mask tokens.
		If masking is "bert", Replace 80% of indices with mask tokens, 
							  10% with the original token, 
							  and 10% with a random token.
	
		params:
			inputs (torch.Tensor)			: a tensor containing token ids for a model
			tokenizer (PreTrainedTokenizer)	: the tokenizer for the model for which the inputs are being prepared
			indices (list[int])				: which positions to mask in the input
											  if no indices are passed and masking is in ['bert', 'roberta', 'none'],
											  a random 15% of tokens will be chosen to mask
			masking_style (str)				: a string specifying which masked tuning style to use
											  'always' means always replace the indices with the mask
											  'none' means leave the original indices in place
											  'bert/roberta' means replace the indices with the mask 80% of the time,
											  	with the original token 10% of the time,
											  	and with a random token id 10% of the time
			device (str)					: what device (cpu, cuda) to place any returned indices on
		
		returns:
			masked_inputs (torch.Tensor)	: inputs where the indices have been replaced with the mask token id
											  according to the masked tuning style
			indices (List[int])				: if no indices were passed, a list of the randomly chosen indices 
											  to mask is returned 
	'''
	return_indices = False
	if indices is None:
		return_indices = True
		# exclude the pad tokens
		candidates 	= (inputs != tokenizer.convert_tokens_to_ids(tokenizer.pad_token)).nonzero(as_tuple=True)[0]
		indices 	= torch.argsort(torch.rand(candidates.shape[0], device=device))[:round(candidates.shape[0]*.15)]
		# this is just for presentational purposes really
		indices 	= indices.sort().values
	
	# if we are not masking anything just return it
	if masking_style == 'none':
		if return_indices:
			return inputs, candidates
		else:
			return inputs
	
	masked_inputs 	= inputs.clone().detach()	
	mask_token_id 	= tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
	
	for index in indices:
		# even when using bert/roberta style tuning, we sometimes need access to the data with everything masked
		# do not use bert/roberta-style masking if we are always masking
		# note that we DO want to allow collecting unmasked inputs even when using always masked tuning, since we need them for the labels
		# setting this to 0 means we always mask if masking_style is none
		r = np.random.random() if masking_style in ['bert', 'roberta'] else 0
			
		# Roberta tuning regimen: 
		# masked tokens are masked 80% of the time,
		# original 10% of the time, 
		# and random word 10% of the time
		if r < 0.8:
			replacement = mask_token_id
		elif 0.8 <= r < 0.9:
			replacement = inputs[index]
		elif 0.9 <= r:
			replacement = np.random.choice(len(tokenizer))
		
		masked_inputs[index] = replacement
	
	if return_indices:
		return masked_inputs, indices
	else:
		return masked_inputs

def sem(x: Union[List,np.ndarray,torch.Tensor]) -> float:
	'''
	Calculate the standard error of the mean for a list of numbers
	
		params:
			x (list) 		: a list of numbers for which to calculate the standard error of the mean
		
		returns:
			sem_x (float)	: the standard error of the mean of x
	'''
	namespace = torch if isinstance(x,torch.Tensor) else np
	return namespace.std(x)/sqrt(len(x))

class KLDivergenceComparison(KLDivLoss):
	'''
	Gets the KL divergence between the predictions of two models. Higher KL divergences
	mean that the q model's predictions deviate more from the p model's.
	
	Adapted from torch.nn.modules.loss.KLDivLoss
	'''
	def __init__(
		self,
		p_model: 'PreTrainedModel',
		q_model: 'PreTrainedModel',
		dataset: Dataset,
		n_examples: int = None,
		masking: str = 'none',
		p_model_kwargs: Dict = None,
		q_model_kwargs: Dict = None,
		p_tokenizer_kwargs: Dict = None,
		q_tokenizer_kwargs: Dict = None,
		size_average = None,
		reduce = None,
		reduction: str = 'batchmean',
		device: str = 'cpu',
	) -> None:
		'''
		Creates a KL divergence loss object that codes divergence of a fine-tuned
		model compared to a baseline model on a specified dataset.
		
			params:
				model (PreTrainedModel)				: a huggingface pretrained model
				tokenizer (PreTrainedTokenizer)		: a huggingface pretrained tokenizer (should match the model)
				dataset (Dataset)					: a dataset in huggingface's datasets format that
													  has been pretokenized for use with the same kind of tokenizer
													  as passed
				n_examples_per_step (int)			: it may be too time-consuming to calculate the KLBaselineLoss on
													  the basis of the entire dataset, if the dataset is large.
													  you can use this to set how many random samples to draw
													  from dataset to use when calculating loss. If not set,
													  all examples will be used each time.
				masking (str)						: which style of masking to use when calculating the loss
													  valid options are the following.
													  "always": choose 15% of tokens to randomly mask per sentence, and replace with mask tokens
													  "bert"  : choose 15% of tokens to randomly mask per sentence. replace 80% with mask token, 10% with original token, 10% with random word.
													  "none"  : don't mask any tokens. KL divergence is calculated on the full sentence instead of 15% of randomly masked tokens
				model_kwargs (dict)					: used to create a baseline version of the passed model
				tokenizer_kwargs (dict)				: used to create a baseline version of the passed tokenizer
				size_average						: passed to KLDivLoss
				reduce 								: passed to KLDivLoss
				reduction							: passed to KLDivLoss
				device (str)						: what device to put the models on
		'''
		super(KLBaselineLoss, self).__init__(size_average, reduce, reduction)
		
		self.device 		= device if torch.cuda.is_available() else 'cpu'
		self.masking 		= masking
		
		p_model_kwargs 		= {} if p_model_kwargs is None else p_model_kwargs
		q_model_kwargs 		= {} if q_model_kwargs is None else q_model_kwargs
		
		p_tokenizer_kwargs 	= {} if p_tokenizer_kwargs is None else p_tokenizer_kwargs
		q_tokenizer_kwargs 	= {} if q_tokenizer_kwargs is None else q_tokenizer_kwargs
		
		log.info(f'Initializing Comparison Model for KL Divergence:\t{q_model if isinstance(q_model,str) else q_model.name_or_path}')
		self.q_model 		= q_model if not isinstance(q_model,str) else AutoModelForMaskedLM.from_pretrained(q_model, **q_model_kwargs).to(self.device)
		self.q_tokenizer 	= AutoTokenizer.from_pretrained(self.q_model.name_or_path, **q_tokenizer_kwargs)
		
		log.info(f'Initializing Baseline Model for KL Divergence:\t{p_model if isinstance(p_model,str) else p_model.name_or_path}')
		self.p_model		= p_model if not isinstance(p_model,str) else AutoModelForMaskedLM.from_pretrained(p_model, **p_model_kwargs).to(self.device)
		self.p_tokenizer 	= AutoTokenizer.from_pretrained(self.p_model.name_or_path, **p_tokenizer_kwargs)
		
		if not self.p_tokenizer.get_vocab() == self.q_tokenizer.get_vocab():
			raise ValueError('Models must have identical vocabularies!')
		
		# we're not evaluating these
		# _ = is to prevent printing
		_ = self.p_model.eval()
		_ = self.q_model.eval()
		
		self.dataset 		= dataset
		
		# can't use more examples than we've got
		self.n_examples 	= self.dataset.num_rows if n_examples is None else min(n_examples, self.dataset.num_rows)		
	
	def compute(self) -> Tuple[torch.Tensor]:
		'''
		Computes KLBaselineLoss between the predictions of the baseline model
		and the predictions of the fine-tuned model on the basis of self.n_examples
		from self.dataset. Samples are randomized with each call.
		
			params:
				progress_bar (bool)		: whether to display a progress bar while iterating through
										  the chosen examples
				return_all (bool)		: whether to return a list containing every individual KL divergence
										  in a list in addition to the mean
			
			returns:
				kl_div (torch.Tensor)	: the mean KL divergence between the model and the baseline model
										  across n_examples of the dataset, multiplied by the scaling factor
				kl_divs (torch.Tensor)	: the individual KL divergence for each example
										  returned if return_all=True.
		'''
		# construct a comparison dataset for this call with n random examples
		comp_dataset 		= sample_from_dataset(self.dataset, self.n_examples, log_message=progress_bar)
		dataloader 			= tqdm(torch.utils.data.DataLoader(comp_dataset, batch_size=1))
		mean_kl_div			= torch.tensor((0.)).to(self.device)
		kl_divs 			= []
		all_mask_indices	= []
		
		with torch.no_grad():
			for i, batch in enumerate(dataloader):
				batch_inputs	= {k: v.to(self.device) for k, v in batch.items() if isinstance(v,torch.Tensor)}
				
				mask_indices	= []
				for i, _ in enumerate(batch_inputs['input_ids']):
					batch_inputs['input_ids'][i], mask_input_indices \
								= mask_input(
									inputs=batch_inputs['input_ids'][i],
									tokenizer=self.p_tokenizer,
									masking_style=self.masking,
									device=self.device,
								)
					
					mask_indices.append(mask_input_indices)
				
				if return_all:
					all_mask_indices.append(mask_indices)
				
				outputs 		= self.q_model(**batch_inputs).logits
				outputs 		= F.log_softmax(outputs, dim=-1)
				p_outputs 		= F.softmax(self.p_model(**batch_inputs).logits, dim=-1)	
								
				# we just calculate the loss on the selected tokens
				outputs 		= torch.cat([torch.unsqueeze(outputs[i].index_select(0, mask_locations), dim=0) for i, mask_locations in enumerate(mask_indices)], dim=0)
				p_outputs		= torch.cat([torch.unsqueeze(p_outputs[i].index_select(0, mask_locations), dim=0) for i, mask_locations in enumerate(mask_indices)], dim=0)
				
				# we want this to be a mean instead of a sum, so divide by the length of the dataset
				kl_div 			= super(KLBaselineLoss, self).forward(outputs, p_outputs)
				mean_kl_div 	+= kl_div/comp_dataset.num_rows
				
				kl_divs.append(kl_div.cpu())
				dataloader.set_postfix(kl_div_mean=f'{np.mean(kl_divs):.2f}', kl_div_se=f'{sem(kl_divs):.2f}')
		
		return torch.tensor(kl_divs).to(self.device), all_mask_indices

@hydra.main(config_path='.', config_name='kl_comparison')
def main(cfg: DictConfig):
	configs = os.listdir('model')
	paired = list(permutations(configs,2))
	results = []
	
	for f1, f2 in paired:
		c1 = OmegaConf.load(f1)
		c2 = OmegaConf.load(f2)
	
		dataset = load_format_dataset(
					dataset_loc=cfg.dataset_loc, 
					split=cfg.split, 
					data_field=cfg.data_field, 
					string_id=c1.string_id, 
					n_examples=cfg.n_examples if not str(cfg.n_examples.lower()) == 'none' else None, 
					tokenizer_kwargs=c1.tokenizer_kwargs,
				)
	
		kl_comp = KLDivergenceComparison(
			p_model=c1.string_id, 
			q_model=c2.string_id,
			dataset=dataset,
			n_examples=cfg.n_examples if not str(cfg.n_examples.lower()) == 'none' else None,
			masking=cfg.kl_masking,
			p_model_kwargs=c1.model_kwargs,
			q_model_kwargs=c2.model_kwargs,
			p_tokenizer_kwargs=c1.tokenizer_kwargs,
			q_tokenizer_kwargs=c2.tokenizer_kwargs,
			size_average=cfg.size_average if not cfg.size_average.lower() == 'none' else None,
			reduce=cfg.reduce if not cfg.reduce.lower() == 'none' else None,
			reduction=cfg.reduction,
			device=cfg.device if torch.cuda.is_available() else 'cpu'		
		)
		
		_, kl_divs, all_mask_indices = kl_comp.compute()
		
		pair_results = [dict(
			sentence=sentence,
			kl_div=kl_div,
			mask_indices=mask_indices,
			p_model=c1.string_id,
			q_model=c2.string_id,
			dataset=cfg.dataset_loc,
			split=cfg.split,
			n_examples=cfg.n_examples,
			kl_masking=cfg.kl_masking,
			size_average=cfg.size_average,
			reduce=cfg.reduce,
			reduction=cfg.reduction,
		) for sentence, kl_div, mask_indices in zip(dataset[cfg.data_field], kl_divs, all_mask_indices)]
		
		results.extend(pair_results)
	
	results = pd.DataFrame(results).assign(run_id=os.path.split(os.getcwd())[1])
	results.to_csv('results.csv.gz', index=False, na_rep='NaN')
		
if __name__ == '__main__':
	
	main()