""" Common functions for package biobb_ml.utils """
import matplotlib.pyplot as plt
import csv
import re
from pathlib import Path, PurePath
from importlib import import_module
from biobb_common.tools import file_utils as fu
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


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

def getIndependentVarsList(independent_vars):
    if 'indexes' in independent_vars:
        return ', '.join(str(x) for x in independent_vars['indexes'])
    elif 'range' in independent_vars:
        return ', '.join([str(y) for r in independent_vars['range'] for y in range(r[0], r[1] + 1)])
    elif 'columns' in independent_vars:
        return ', '.join(independent_vars['columns'])

def getTargetsList(targets, tool, out_log, classname):

    if not targets and tool == 'drop': 
    	fu.log(classname + ': No targets provided, exiting', out_log)
    	raise SystemExit(classname + ': No targets provided, exiting')
    elif not targets and tool != 'drop':
    	fu.log('No targets provided, all columns will be taken', out_log)

    if 'indexes' in targets:
        return targets['indexes']
    elif 'range' in targets:
        return ([y for r in targets['range'] for y in range(r[0], r[1] + 1)])
    elif 'columns' in targets:
        return targets['columns']

def getHeader(file):
    with open(file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    if(len(header) == 1):
        return list(re.sub('\s+|;|:|,|\t', ',', header[0]).split(","))
    else:
        return header