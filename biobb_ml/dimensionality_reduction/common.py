""" Common functions for package biobb_analysis.ambertools """
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import csv
import re
from biobb_common.tools import file_utils as fu
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
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
        'output_plot_path': ['png']
	}
	return ext in formats[argument]

# UTILITIES

def getWindowLength(default, feat):
    window_length = default
    # if features size is less than WL, then get last odd
    if feat < window_length: 
        if (feat % 2) == 0:
            window_length = feat - 1
        else:
            window_length = feat
    return window_length

def generate_columns_labels(label, length):
    return [label + ' ' + str(x + 1) for x in range(0, length)]

def plot2D(ax, pca_table, targets, target, x, y):
    ax.set_xlabel('PC ' + str(x), fontsize = 12)
    ax.set_ylabel('PC ' + str(y), fontsize = 12)
    ax.set_title('2 Component PCA (PC ' + str(x) + ' vs PC ' + str(y) + ')', fontsize = 15)
    
    colors = plt.get_cmap('rainbow_r')(np.linspace(0.0, 1.0, len(targets)))
    for tgt, color in zip(targets,colors):
        indicesToKeep = pca_table[target] == tgt
        ax.scatter(pca_table.loc[indicesToKeep, 'PC ' + str(x)]
                   , pca_table.loc[indicesToKeep, 'PC ' + str(y)]
                   , color = color
                   , s = 50
                   , alpha = 0.6)
    if len(targets) < 15: ax.legend(targets)


def PCA2CPlot(pca_table, targets, target):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    plot2D(ax,pca_table, targets, target, 1, 2)
    plt.tight_layout()

def scatter3DLegend(targets):
    colors = plt.get_cmap('rainbow_r')(np.linspace(0.0, 1.0, len(targets)))
    proxies = []
    for i, v in enumerate(targets):
        proxies.append(Line2D([0],[0], linestyle="none", c=colors[i], marker = 'o'))
    return proxies

def plot3D(ax, pca_table, targets, dt):
    xs = pca_table['PC 1']
    ys = pca_table['PC 2']
    zs = pca_table['PC 3']
    ax.scatter(xs, ys, zs, s=50, alpha=0.6, c=dt,cmap='rainbow_r')

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    if len(targets) < 15: 
        scatter_proxies = scatter3DLegend(targets)
        ax.legend(scatter_proxies, targets, numpoints = 1)

    plt.title('3 Component PCA', size=15, pad=35)

def PCA3CPlot(pca_table, targets, target):
    lst = pca_table[target].unique().tolist()
    dct = {lst[i]: i for i in range(0, len(lst))} 
    dt = pca_table[target].map(dct)

    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(2,2,1, projection='3d') 
    
    plot3D(ax, pca_table, targets, dt)

    ax = fig.add_subplot(2,2,2) 
    
    plot2D(ax,pca_table, targets, target, 1, 2)

    ax = fig.add_subplot(2,2,3) 
    
    plot2D(ax,pca_table, targets, target, 1, 3)

    ax = fig.add_subplot(2,2,4) 
    
    plot2D(ax,pca_table, targets, target, 2, 3)

    plt.tight_layout()

def predictionPlot(tit, data1, data2, xlabel, ylabel):   
    z = np.polyfit(data1, data2, 1)
    plt.scatter(data2, data1, alpha=0.2)
    plt.title(tit, size=15)
    plt.xlabel(xlabel,size=14)
    plt.ylabel(ylabel,size=14)
    #Plot the best fit line
    plt.plot(np.polyval(z,data1), data1, c='red', linewidth=1)
    #Plot the ideal 1:1 line
    axes = plt.gca()
    lims = axes.get_xlim()
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.legend(('Best fit','Ideal 1:1'))

def histogramPlot(tit, data1, data2, xlabel, ylabel):
    plt.title(tit, size=15)
    error = data2 - data1
    plt.hist(error, bins = 25)
    plt.xlabel(xlabel,size=14)
    plt.ylabel(ylabel,size=14)

def PLSRegPlot(y, y_c, y_cv):

    #FIGURE
    plt.figure(figsize=[8,8])

    plt.subplot(221)
    predictionPlot('Calibration predictions', y, y_c, 'true values', 'predictions')

    plt.subplot(222)
    histogramPlot('Calibration histogram', y, y_c[0], 'prediction error', 'count')

    plt.subplot(223)
    predictionPlot('Cross Validation predictions', y, y_cv, 'true values', 'predictions')

    plt.subplot(224)
    histogramPlot('Cross Validation histogram', y, y_cv[0], 'prediction error', 'count')

    plt.tight_layout()
    
    return plt

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

def getTarget(target, data, out_log, classname):
    if 'index' in target:
        return data.iloc[:, target['index']]
    elif 'column' in target:
        return data[target['column']]
    else:
        fu.log(classname + ': Incorrect target format', out_log)
        raise SystemExit(classname + ': Incorrect target format')

def getTargetValue(target):
    if 'index' in target:
        return str(target['index'])
    elif 'column' in target:
        return target['column']

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