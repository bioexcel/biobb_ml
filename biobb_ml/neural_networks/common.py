""" Common functions for package biobb_analysis.ambertools """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import csv
import re
from pathlib import Path, PurePath
from sklearn.metrics import roc_curve, auc
from biobb_common.tools import file_utils as fu
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
sns.set()

# CHECK PARAMETERS

def check_input_path(path, argument, optional, out_log, classname):
    """ Checks input file """ 
    if optional and not path:
        return None
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
        'input_decode_path': ['csv'],
        'input_predict_path': ['csv'],
        'input_model_path': ['h5'],
        'output_model_path': ['h5'],
		'output_results_path': ['csv'],
		'output_test_table_path': ['csv'],
        'output_test_decode_path': ['csv'],
        'output_test_predict_path': ['csv'],
        'output_decode_path': ['csv'],
        'output_predict_path': ['csv'],
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

def get_keys_of_predictors(predictions):
    p = []
    for obj in predictions[0]:
        p.append(obj)
    return p

def get_num_cols(num):
    p = []
    for i in range(1, num + 1):
        p.append('item ' + str(i))
    return p

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.asarray(X), np.asarray(y)

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

def CMPlotBinary(position, cm, group_names, title, normalize):
    plt.subplot(position)
    plt.title(title, size=15)
    if normalize:
        group_counts = ["{0:0.2f}".format(value) for value in cm.flatten()]
    else:
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
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

def plotResultsClassMultCM(data, cm_train, cm_test, normalize, values):

	#FIGURE
    plt.figure(figsize=[12,8])

    plt.subplot(231)
    doublePlot('Model loss', data['loss'], data['val_loss'], 'epoch', 'loss', ['training', 'validation'])

    plt.subplot(232)
    doublePlot('Model accuracy', data['accuracy'], data['val_accuracy'], 'epoch', 'accuracy', ['training', 'validation'])

    plt.subplot(233)
    doublePlot('Model MSE', data['mse'], data['val_mse'], 'epoch', 'mse', ['training', 'validation'])

    CMplotNonBinary(234, cm_train, 'Confusion Matrix Train', normalize, values)

    CMplotNonBinary(235, cm_test, 'Confusion Matrix Test', normalize, values)

    plt.tight_layout()
    
    return plt

def distPredPlot(position, y, pos_p, labels, title):
    df = pd.DataFrame({'probPos':pos_p, 'target': y})
    plt.subplot(position)
    plt.hist(df[df.target==1].probPos, density=True, bins=25,
             alpha=.5, color='green',  label=labels[0])
    plt.hist(df[df.target==0].probPos, density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0,1])
    plt.title(title, size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")

def ROCPlot(position, y, p, cm, title):
    fp_rates, tp_rates, _ = roc_curve(y,p)
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(position)
    plt.plot(fp_rates, tp_rates, color='green',
             lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    #plot current decision point:
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title(title, size=15)
    plt.legend(loc="lower right")

def plotResultsClassBinCM(data, proba_train, proba_test, y_train, y_test, cm_train, cm_test, normalize, values):

	#FIGURE
    plt.figure(figsize=[15,15])

    plt.subplot(331)
    doublePlot('Model loss', data['loss'], data['val_loss'], 'epoch', 'loss', ['training', 'validation'])

    plt.subplot(332)
    doublePlot('Model accuracy', data['accuracy'], data['val_accuracy'], 'epoch', 'accuracy', ['training', 'validation'])

    plt.subplot(333)
    doublePlot('Model MSE', data['mse'], data['val_mse'], 'epoch', 'mse', ['training', 'validation'])

    pos_p = proba_train[:,1]

    #CMplotNonBinary(334, cm_train, 'Confusion Matrix Train', normalize, values)
    CMPlotBinary(334, cm_train, ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'], 'Confusion Matrix Train', normalize)

    distPredPlot(335, y_train, pos_p, ['Positives', 'Negatives'], 'Distributions of Predictions Train')

    ROCPlot(336, y_train, pos_p, cm_train, 'ROC Curve Train')

    pos_p = proba_test[:,1]

    #CMplotNonBinary(337, cm_test, 'Confusion Matrix Test', normalize, values)
    CMPlotBinary(337, cm_test, ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'], 'Confusion Matrix Test', normalize)

    distPredPlot(338, y_test, pos_p, ['Positives', 'Negatives'], 'Distributions of Predictions Test')

    ROCPlot(339, y_test, pos_p, cm_test, 'ROC Curve Test')

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

def getFeatures(independent_vars, data, out_log, classname):
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

def getTarget(target, data, out_log, classname):
    if 'index' in target:
        return data.iloc[:, target['index']]
    elif 'column' in target:
        return data[target['column']]
    else:
        fu.log(classname + ': Incorrect target format', out_log)
        raise SystemExit(classname + ': Incorrect target format')

def getWeight(weight, data, out_log, classname):
    if 'index' in weight:
        return data.iloc[:, weight['index']]
    elif 'column' in weight:
        return data[weight['column']]
    else:
        fu.log(classname + ': Incorrect weight format', out_log)
        raise SystemExit(classname + ': Incorrect weight format')

def getHeader(file):
    with open(file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

    if(len(header) == 1):
        return list(re.sub('\s+|;|:|,|\t', ',', header[0]).split(","))
    else:
        return header
        
def getTargetValue(target):
    if 'index' in target:
        return target['index']
    elif 'column' in target:
        return target['column']