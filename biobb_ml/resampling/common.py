""" Common functions for package biobb_ml.resampling """
from pathlib import Path, PurePath
from importlib import import_module
from biobb_common.tools import file_utils as fu
import warnings
import csv
import re
import pandas as pd

# UNDERSAMPLING METHODS
undersampling_methods = {
	'random':{
		'method': 'RandomUnderSampler',
		'module': 'imblearn.under_sampling'
	},
	'nearmiss':{
		'method': 'NearMiss',
		'module': 'imblearn.under_sampling'
	},
	'cnn':{
		'method': 'CondensedNearestNeighbour',
		'module': 'imblearn.under_sampling'
	},
	'tomeklinks':{
		'method': 'TomekLinks',
		'module': 'imblearn.under_sampling'
	},
	'enn':{
		'method': 'EditedNearestNeighbours',
		'module': 'imblearn.under_sampling'
	},
	'ncr':{
		'method': 'NeighbourhoodCleaningRule',
		'module': 'imblearn.under_sampling'
	},
	'cluster':{
		'method': 'ClusterCentroids',
		'module': 'imblearn.under_sampling'
	}
}

# OVERSAMPLING METHODS
oversampling_methods = {
	'random':{
		'method': 'RandomOverSampler',
		'module': 'imblearn.over_sampling'
	},
	'smote':{
		'method': 'SMOTE',
		'module': 'imblearn.over_sampling'
	},
	'borderline':{
		'method': 'BorderlineSMOTE',
		'module': 'imblearn.over_sampling'
	},
	'svmsmote':{
		'method': 'SVMSMOTE',
		'module': 'imblearn.over_sampling'	
	},
	'adasyn':{
		'method': 'ADASYN',
		'module': 'imblearn.over_sampling'	
	}
}

# RESAMPLING METHODS
resampling_methods = {
	'smotetomek':{
		'method': 'SMOTETomek',
		'module': 'imblearn.combine',
		'method_over': 'SMOTE',
		'module_over': 'imblearn.over_sampling',
		'method_under': 'TomekLinks',
		'module_under': 'imblearn.under_sampling'
	},
	'smotenn':{
		'method': 'SMOTEENN',
		'module': 'imblearn.combine',
		'method_over': 'SMOTE',
		'module_over': 'imblearn.over_sampling',
		'method_under': 'EditedNearestNeighbours',
		'module_under': 'imblearn.under_sampling'
	}
}

# CHECK PARAMETERS

def check_input_path(path, argument, out_log, classname):
	""" Checks input file """ 
	if not Path(path).exists():
		fu.log(classname + ': Unexisting %s file, exiting' % argument, out_log)
		raise SystemExit(classname + ': Unexisting %s file' % argument)
	file_extension = PurePath(path).suffix
	if not is_valid_file(file_extension[1:], argument):
		fu.log(classname + ': Format %s in %s file is not compatible' % (file_extension[1:], argument), out_log)
		raise SystemExit(classname + ': Format %s in %s file is not compatible' % (file_extension[1:], argument))
	return path

def check_output_path(path, argument, optional, out_log, classname):
	""" Checks output file """ 
	if optional and not path:
		return None
	if PurePath(path).parent and not Path(PurePath(path).parent).exists():
		fu.log(classname + ': Unexisting  %s folder, exiting' % argument, out_log)
		raise SystemExit(classname + ': Unexisting  %s folder' % argument)
	file_extension = PurePath(path).suffix
	if not is_valid_file(file_extension[1:], argument):
		fu.log(classname + ': Format %s in  %s file is not compatible' % (file_extension[1:], argument), out_log)
		raise SystemExit(classname + ': Format %s in  %s file is not compatible' % (file_extension[1:], argument))
	return path

def is_valid_file(ext, argument):
	""" Checks if file format is compatible """
	formats = {
		'input_dataset_path': ['csv', 'txt'],
		'output_dataset_path': ['csv'],
		'output_plot_path': ['png'],
		'input_model_path': ['pkl']
	}
	return ext in formats[argument]

def getTarget(target, data, out_log, classname):
	""" Gets targets """
	if 'index' in target:
		return data.iloc[:, target['index']]
	elif 'column' in target:
		return data[target['column']]
	else:
		fu.log(classname + ': Incorrect target format', out_log)
		raise SystemExit(classname + ': Incorrect target format')

def getTargetValue(target, out_log, classname):
	""" Gets target value """
	if 'index' in target:
		return target['index']
	elif 'column' in target:
		return target['column']
	else:
		fu.log(classname + ': Incorrect target format', out_log)
		raise SystemExit(classname + ': Incorrect target format')

def getHeader(file):
	
	with open(file, newline='') as f:
		reader = csv.reader(f)
		header = next(reader)
        
	if(len(header) == 1):
		return list(re.sub('\s+|;|:|,|\t', ',', header[0]).split(","))
	else:
		return header


def checkResamplingType(type_, out_log, classname):
	""" Gets resampling type """
	if not type_:
		fu.log(classname + ': Missed mandatory type property', out_log)
		raise SystemExit(classname + ': Missed mandatory type property')
	if type_ != 'regression' and type_ != 'classification':
		fu.log(classname + ': Unknown %s type property' % type_, out_log)
		raise SystemExit(classname + ': Unknown %s type property' % type_)


def getResamplingMethod(method, type_, out_log, classname):
	""" Gets resampling method """
	if type_ == 'undersampling': methods = undersampling_methods
	elif type_ == 'oversampling': methods = oversampling_methods
	elif type_ == 'resampling': methods = resampling_methods

	if not method:
		fu.log(classname + ': Missed mandatory method property', out_log)
		raise SystemExit(classname + ': Missed mandatory method property')
	if not method in methods:
		fu.log(classname + ': Unknown %s method property' % method, out_log)
		raise SystemExit(classname + ': Unknown %s method property' % method)

	mod = import_module(methods[method]['module'])
	warnings.filterwarnings("ignore")
	method_to_call = getattr(mod, methods[method]['method'])

	fu.log('%s method selected' % methods[method]['method'], out_log)
	return method_to_call

def getCombinedMethod(method, out_log, classname):
	""" Gets combinded method """
	methods = resampling_methods

	if not method:
		fu.log(classname + ': Missed mandatory method property', out_log)
		raise SystemExit(classname + ': Missed mandatory method property')
	if not method in methods:
		fu.log(classname + ': Unknown %s method property' % method, out_log)
		raise SystemExit(classname + ': Unknown %s method property' % method)

	mod = import_module(methods[method]['module'])
	warnings.filterwarnings("ignore")
	method_to_call = getattr(mod, methods[method]['method'])

	fu.log('%s method selected' % methods[method]['method'], out_log)

	mod_over = import_module(methods[method]['module_over'])
	method_over_to_call = getattr(mod_over, methods[method]['method_over'])
	mod_under = import_module(methods[method]['module_under'])
	method_under_to_call = getattr(mod_under, methods[method]['method_under'])

	return method_to_call, method_over_to_call, method_under_to_call

def getSamplingStrategy(sampling_strategy, out_log, classname):
	""" Gets sampling strategy """
	if 'target' in sampling_strategy:
		if isinstance(sampling_strategy['target'], str):
			return sampling_strategy['target']
	if 'ratio' in sampling_strategy:
		if isinstance(sampling_strategy['ratio'], float) and sampling_strategy['ratio'] >= 0 and sampling_strategy['ratio'] <= 1:
			return sampling_strategy['ratio']
	if 'dict' in sampling_strategy:
		if isinstance(sampling_strategy['dict'], dict):
			# trick for ensure the keys are integers
			samp_str = {}
			for key, item in sampling_strategy['dict'].items():
				samp_str[int(key)] = item
			return samp_str
	if 'list' in sampling_strategy:
		if isinstance(sampling_strategy['list'], list):
			return sampling_strategy['list']

	fu.log(classname + ': Incorrect sampling_strategy format', out_log)
	raise SystemExit(classname + ': Incorrect sampling_strategy format')


