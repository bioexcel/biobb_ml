""" Common functions for package biobb_analysis.ambertools """
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
from importlib import import_module
from biobb_common.tools import file_utils as fu
import warnings

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

def check_mandatory_property(property, name, out_log, classname):
	""" Checks if property is mandatory """
	if not property:
		fu.log(classname + ': Unexisting  %s property, exiting' % name, out_log)
		raise SystemExit(classname + ': Unexisting  %s property' % name)
	return property

def getIndependentVars(independent_vars, data, out_log, classname):
	""" Gets independent vars """
	if 'indexes' in independent_vars:
		return data.iloc[:, independent_vars['indexes']]
	elif 'range' in independent_vars:
		ranges_list = []
		for rng in independent_vars['range']:
			for x in range (rng[0], (rng[1] + 1)):
				ranges_list.append(x)
		return data.iloc[:, ranges_list]
	elif 'columns' in independent_vars:
		return data.loc[:, independent_vars['columns']]
	else:
		fu.log(classname + ': Incorrect independent_vars format', out_log)
		raise SystemExit(classname + ': Incorrect independent_vars format')

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

def checkResamplingType(type_, out_log, classname):
	""" Gets resampling type """
	if not type_:
		fu.log(classname + ': Missed mandatory type property', out_log)
		raise SystemExit(classname + ': Missed mandatory type property')
	if type_ != 'regression' and type_ != 'classification':
		fu.log(classname + ': Unknown %s type property' % type_, out_log)
		raise SystemExit(classname + ': Unknown %s type property' % type_)


def getResamplingMethod(method, type_, out_log, classname):
	""" Gets undersampling method """
	if type_ == 'undersampling': methods = undersampling_methods
	else: methods = oversampling_methods

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


