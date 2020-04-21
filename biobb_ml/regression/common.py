""" Common functions for package biobb_analysis.ambertools """
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
from biobb_common.tools import file_utils as fu

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
		'input_dataset_path': ['csv'],
		'output_dataset_path': ['csv'],
		'output_results_path': ['csv'],
		'output_test_table_path': ['csv'],
		'output_plot_path': ['png']
	}
	return ext in formats[argument]

def check_mandatory_property(property, name, out_log, classname):
	if not property:
		fu.log(classname + ': Unexisting  %s property, exiting' % name, out_log)
		raise SystemExit(classname + ': Unexisting  %s property' % name)
	return property


# UTILITIES

def adjusted_r2(x, y, r2):
    n = x.shape[0]
    p = x.shape[1]

    return 1-(1-r2)*(n-1)/(n-p-1)

def get_list_of_predictors(predictions):
	p = []
	for obj in predictions:
		a = []
		for k, v in obj.items():
			a.append(v)
		p.append(a)
	return p

def plotResults(y_train, y_hat_train, y_test, y_hat_test):

	#FIGURE
    plt.figure(figsize=[12,5])

    plt.subplot(121)
    plt.title('Train', size=15)
    plt.scatter(y_train, y_hat_train, alpha=0.2)
    plt.xlabel('targets',size=18)
    plt.ylabel('predictions',size=18)

    plt.subplot(122)
    plt.title('Test', size=15)
    plt.scatter(y_test, y_hat_test, alpha=0.2)
    plt.xlabel('targets',size=18)
    plt.ylabel('predictions',size=18)

    plt.subplots_adjust(wspace=.3, hspace=.3)
    
    return plt
