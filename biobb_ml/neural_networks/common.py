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

def get_list_of_predictors(predictions):
	p = []
	for obj in predictions:
		a = []
		for k, v in obj.items():
			a.append(v)
		p.append(a)
	return p

def plotResults(data):

	#FIGURE
    plt.figure(figsize=[12,4])

    plt.subplot(131)
    plt.title('Model loss', size=15)
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.xlabel('loss',size=14)
    plt.ylabel('epoch',size=14)
    plt.legend(['training', 'validation'], loc='best')

    plt.subplot(132)
    plt.title('Model accuracy', size=15)
    plt.plot(data['accuracy'])
    plt.plot(data['val_accuracy'])
    plt.xlabel('accuracy',size=14)
    plt.ylabel('epoch',size=14)
    plt.legend(['training', 'validation'], loc='best')

    plt.subplot(133)
    plt.title('Model MSE', size=15)
    plt.plot(data['mse'])
    plt.plot(data['val_mse'])
    plt.xlabel('mse',size=14)
    plt.ylabel('epoch',size=14)
    plt.legend(['training', 'validation'], loc='best')

    plt.tight_layout()
    
    return plt