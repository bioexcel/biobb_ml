""" Common functions for package biobb_analysis.ambertools """
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
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
		'output_model_path': ['pkl'],
		'output_dataset_path': ['csv'],
		'output_results_path': ['csv'],
		'input_model_path': ['pkl'],
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

def get_keys_of_predictors(predictions):
    p = []
    for obj in predictions[0]:
        p.append(obj)
    return p

def predictionPlot(tit, data1, data2, xlabel, ylabel):
    plt.title(tit, size=15)
    plt.scatter(data1, data2, alpha=0.2)
    plt.xlabel(xlabel,size=14)
    plt.ylabel(ylabel,size=14)
    axes = plt.gca()
    lims = axes.get_xlim()
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)

def histogramPlot(tit, data1, data2, xlabel, ylabel):
    plt.title(tit, size=15)
    error = data2 - data1
    plt.hist(error, bins = 25)
    plt.xlabel(xlabel,size=14)
    plt.ylabel(ylabel,size=14)

def plotResults(y_train, y_hat_train, y_test, y_hat_test):

	#FIGURE
    plt.figure(figsize=[8,8])

    plt.subplot(221)
    predictionPlot('Train predictions', y_train, y_hat_train, 'true values', 'predictions')

    plt.subplot(222)
    histogramPlot('Train histogram', y_train, y_hat_train, 'prediction error', 'count')

    plt.subplot(223)
    predictionPlot('Test predictions', y_test, y_hat_test, 'true values', 'predictions')

    plt.subplot(224)
    histogramPlot('Test histogram', y_test, y_hat_test, 'prediction error', 'count')

    plt.tight_layout()
    
    return plt
