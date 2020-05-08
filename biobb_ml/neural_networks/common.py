""" Common functions for package biobb_analysis.ambertools """
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
from pathlib import Path, PurePath
from biobb_common.tools import file_utils as fu
sns.set()

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

def doublePlot(tit, data1, data2, xlabel, ylabel, legend):
    plt.title(tit, size=15)
    plt.plot(data1)
    plt.plot(data2)
    plt.xlabel(xlabel,size=14)
    plt.ylabel(ylabel,size=14)
    plt.legend(legend, loc='best')

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

def CMPlotBinary(position, cm, group_names, title):
    plt.subplot(position)
    plt.title(title, size=15)
    #group_names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    labels_cfm = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts, group_names)]
    labels_cfm = np.asarray(labels_cfm).reshape(2,2)
    sns.heatmap(cm, annot=labels_cfm, fmt='', cmap='Blues', square=True)
    plt.ylabel('True Values', size=13)
    plt.xlabel('Predicted Values', size=13)
    plt.yticks(rotation=0)

def CMplotNonBinary(position, cm, title, normalize, values):

    if cm.shape[1] < 5:  fs = 10
    elif cm.shape[1] >= 5 and cm.shape[1] < 10:  fs = 8
    elif cm.shape[1] >= 10: fs = 6

    plt.subplot(position)
    plt.title(title, size=15)
    if normalize:
        group_counts = ["{0:0.2f}".format(value) for value in cm.flatten()]
    else:
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_names = []
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j: group_names.append("True " + str(values[i]))
        else: group_names.append("False " + str(values[i]))
    labels_cfm = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_names)]
    labels_cfm = np.asarray(labels_cfm).reshape(cm.shape[0],cm.shape[1])
    sns.heatmap(cm, annot=labels_cfm, fmt='', cmap='Blues', xticklabels=values, yticklabels=values, square=True, annot_kws={"fontsize":fs})
    plt.ylabel('True Values', size=13)
    plt.xlabel('Predicted Values', size=13)
    plt.yticks(rotation=0) 

def plotResultsClass(data, cm_train, cm_test, normalize, values):

	#FIGURE
    plt.figure(figsize=[12,8])

    plt.subplot(231)
    doublePlot('Model loss', data['loss'], data['val_loss'], 'epoch', 'loss', ['training', 'validation'])

    plt.subplot(232)
    doublePlot('Model accuracy', data['accuracy'], data['val_accuracy'], 'epoch', 'accuracy', ['training', 'validation'])

    plt.subplot(233)
    doublePlot('Model MSE', data['mse'], data['val_mse'], 'epoch', 'mse', ['training', 'validation'])

    CMplotNonBinary(234, cm_train, 'Confusion Matrix Train', normalize, values)

    #2 -- Confusion matrix test
    CMplotNonBinary(235, cm_test, 'Confusion Matrix Test', normalize, values)

    plt.tight_layout()
    
    return plt

def plotResultsReg(data, test_labels, test_predictions, train_labels, train_predictions):

	#FIGURE
    plt.figure(figsize=[12,12])

    plt.subplot(331)
    doublePlot('Model loss', data['loss'], data['val_loss'], 'epoch', 'loss', ['training', 'validation'])

    plt.subplot(332)
    doublePlot('Model MAE', data['mae'], data['val_mae'], 'epoch', 'mae', ['training', 'validation'])

    plt.subplot(333)
    doublePlot('Model MSE', data['mse'], data['val_mse'], 'epoch', 'mse', ['training', 'validation'])

    plt.subplot(334)
    predictionPlot('Train predictions', train_labels, train_predictions, 'true values', 'predictions')
    
    plt.subplot(335)
    histogramPlot('Train histogram', train_labels, train_predictions, 'prediction error', 'count')

    plt.subplot(337)
    predictionPlot('Test predictions', test_labels, test_predictions, 'true values', 'predictions')
    
    plt.subplot(338)
    histogramPlot('Test histogram', test_labels, test_predictions, 'prediction error', 'count')

    plt.tight_layout()
    
    return plt