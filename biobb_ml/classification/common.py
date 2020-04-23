""" Common functions for package biobb_analysis.ambertools """
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
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

def plotBinaryClassifier(model, proba_train, proba_test, cm_train, cm_test, y_train, y_test, normalize=False, labels=['Positives','Negatives'], cmticks=[0,1], get_plot = True):
    '''
    Visualize the performance of  a Logistic Regression Binary Classifier.
    https://towardsdatascience.com/how-to-interpret-a-binary-logistic-regressor-with-scikit-learn-6d56c5783b49
    '''

    # TRAINING

    #model predicts probabilities of positive class
    p = proba_train
    if len(model.classes_)!=2:
        raise ValueError('A binary class problem is required')
    if model.classes_[1] == 1:
        pos_p = p[:,1]
    elif model.classes_[0] == 1:
        pos_p = p[:,0]
    
    #FIGURE
    plt.figure(figsize=[15,8])
    
    #1 -- Confusion matrix train
    plt.subplot(231)
    plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix Train', size=15)
    plt.colorbar()
    tick_marks = np.arange(len(cmticks))
    plt.xticks(tick_marks, cmticks)
    plt.yticks(tick_marks, cmticks)

    cmlabels = [
    	['True Negatives', 'False Positives'],
        ['False Negatives', 'True Positives']
    ]

    fmt = '.2f' if normalize else 'd'
    thresh = cm_train.max() / 2.
    for i, j in itertools.product(range(cm_train.shape[0]), range(cm_train.shape[1])):
        plt.text(j, i, format(cm_train[i, j], fmt) + "\n" + cmlabels[i][j],
                 horizontalalignment="center",
                 color="white" if cm_train[i, j] > thresh else "black")

    plt.ylabel('True Values', size=13)
    plt.xlabel('Predicted Values', size=13)
      
    #2 -- Distributions of Predicted Probabilities of both classes train
    df = pd.DataFrame({'probPos':pos_p, 'target': y_train})
    plt.subplot(232)
    plt.hist(df[df.target==1].probPos, density=True, bins=25,
             alpha=.5, color='green',  label=labels[0])
    plt.hist(df[df.target==0].probPos, density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0,1])
    plt.title('Distributions of Predictions Train', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")
    
    #3 -- ROC curve with annotated decision point train
    fp_rates, tp_rates, _ = roc_curve(y_train,p[:,1])
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(233)
    plt.plot(fp_rates, tp_rates, color='green',
             lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    #plot current decision point:
    tn, fp, fn, tp = [i for i in cm_train.ravel()]
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve Train', size=15)
    plt.legend(loc="lower right")

    # TESTING

    #model predicts probabilities of positive class
    p = proba_test
    if len(model.classes_)!=2:
        raise ValueError('A binary class problem is required')
    if model.classes_[1] == 1:
        pos_p = p[:,1]
    elif model.classes_[0] == 1:
        pos_p = p[:,0]
    
    #1 -- Confusion matrix test
    plt.subplot(234)
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix Test', size=15)
    plt.colorbar()
    tick_marks = np.arange(len(cmticks))
    plt.xticks(tick_marks, cmticks)
    plt.yticks(tick_marks, cmticks)

    cmlabels = [
    	['True Negatives', 'False Positives'],
        ['False Negatives', 'True Positives']
    ]

    fmt = '.2f' if normalize else 'd'
    thresh = cm_test.max() / 2.
    for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
        plt.text(j, i, format(cm_test[i, j], fmt) + "\n" + cmlabels[i][j],
                 horizontalalignment="center",
                 color="white" if cm_test[i, j] > thresh else "black")

    plt.ylabel('True Values', size=13)
    plt.xlabel('Predicted Values', size=13)
      
    #2 -- Distributions of Predicted Probabilities of both classes test
    df = pd.DataFrame({'probPos':pos_p, 'target': y_test})
    plt.subplot(235)
    plt.hist(df[df.target==1].probPos, density=True, bins=25,
             alpha=.5, color='green',  label=labels[0])
    plt.hist(df[df.target==0].probPos, density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0,1])
    plt.title('Distributions of Predictions Test', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")
    
    #3 -- ROC curve with annotated decision point test
    fp_rates, tp_rates, _ = roc_curve(y_test,p[:,1])
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(236)
    plt.plot(fp_rates, tp_rates, color='green',
             lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    #plot current decision point:
    tn, fp, fn, tp = [i for i in cm_test.ravel()]
    plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve Test', size=15)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=.3, hspace=.5)
    
    return plt

def plotMultipleCM(cm_train, cm_test, normalize, values):
    
    #FIGURE
    plt.figure(figsize=[8,4])
    
    #1 -- Confusion matrix train
    plt.subplot(121)
    plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix Train', size=15)
    plt.colorbar()
    tick_marks = np.arange(len(values))
    plt.xticks(tick_marks, values)
    plt.yticks(tick_marks, values)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_train.max() / 2.
    for i, j in itertools.product(range(cm_train.shape[0]), range(cm_train.shape[1])):
        if i == j: help_txt = "True " + str(values[i])
        else: help_txt = "False" + str(values[i])
        plt.text(j, i, format(cm_train[i, j], fmt) + "\n" + help_txt,
                 horizontalalignment="center",
                 color="white" if cm_train[i, j] > thresh else "black")

    plt.ylabel('True Values', size=13)
    plt.xlabel('Predicted Values', size=13)

    #2 -- Confusion matrix test
    plt.subplot(122)
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix Test', size=15)
    plt.colorbar()
    tick_marks = np.arange(len(values))
    plt.xticks(tick_marks, values)
    plt.yticks(tick_marks, values)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_test.max() / 2.
    for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
        if i == j: help_txt = "True " + str(values[i])
        else: help_txt = "False" + str(values[i])
        plt.text(j, i, format(cm_test[i, j], fmt) + "\n" + help_txt,
                 horizontalalignment="center",
                 color="white" if cm_test[i, j] > thresh else "black")

    plt.ylabel('True Values', size=13)
    plt.xlabel('Predicted Values', size=13)

    return plt


def get_list_of_predictors(predictions):
	p = []
	for obj in predictions:
		a = []
		for k, v in obj.items():
			a.append(v)
		p.append(a)
	return p
