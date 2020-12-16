""" Common functions for package biobb_analysis.ambertools """
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import csv
import re
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from random import sample
from math import isnan
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
        'output_model_path': ['pkl'],
        'input_model_path': ['pkl'],
		'output_results_path': ['csv'],
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

# get best K in WCSS plot (getting elbow point)
def get_best_K(wcss):
	curve = wcss
	nPoints = len(curve)
	allCoord = np.vstack((range(nPoints), curve)).T
	np.array([range(nPoints), curve])
	firstPoint = allCoord[0]
	lineVec = allCoord[-1] - allCoord[0]
	lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
	vecFromFirst = allCoord - firstPoint
	scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
	vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
	vecToLine = vecFromFirst - vecFromFirstParallel
	distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
	idxOfBestPoint = np.argmax(distToLine)

	return idxOfBestPoint + 1, np.argmax(distToLine)

# hopkins test
# https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(np.random.uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

# compute elbow
def getWCSS(method, max_clusters, t_predictors):
    wcss = []
    for i in range(1,max_clusters + 1):
        if method == 'kmeans': clusterer = KMeans(i)
        elif method == 'agglomerative': clusterer = AgglomerativeClustering(n_clusters=i, linkage="average")
        clusterer.fit(t_predictors)
        wcss_iter = clusterer.inertia_
        wcss.append(wcss_iter)

    return wcss

# compute gap
# https://anaconda.org/milesgranger/gap-statistic/notebook
def getGap(method, data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'cluster':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            clusterer = KMeans(k)
            clusterer.fit(randomReference)
            
            refDisp = clusterer.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        clusterer = KMeans(k)
        clusterer.fit(data)
        
        origDisp = clusterer.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'cluster':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

def getSilhouetthe(method, X, max_clusters, affinity=None, linkage=None, random_state=None):
    # Run clustering with different k and check the metrics
    silhouette_list = []

    k_list = list(range(2,max_clusters + 1))
    for p in k_list:

        if method == 'kmeans': clusterer = KMeans(n_clusters=p, random_state=random_state)
        elif method == 'agglomerative': clusterer = AgglomerativeClustering(n_clusters=p, affinity=affinity, linkage=linkage)
        elif method == 'spectral': clusterer = SpectralClustering(n_clusters=p, affinity = "nearest_neighbors", random_state=random_state)

        clusterer.fit(X)
        # The higher (up to 1) the better
        s = round(silhouette_score(X, clusterer.labels_), 4)

        silhouette_list.append(s)

    k_list.insert(0,1)
    silhouette_list.insert(0,0)

    return silhouette_list, k_list

# plot elbow, gap & silhouette
def plotKmeansTrain(max_clusters, wcss, gap, sil, best_k, best_g, best_s):
    number_clusters = range(1, max_clusters + 1)
    plt.figure(figsize=[15,4])
    #1 -- WCSS
    plt.subplot(131)
    plt.title('The Elbow Method', size=15)
    plt.plot(number_clusters, wcss, '-o')
    plt.axvline(x=best_k, c='red')
    plt.legend(('WCSS', 'Best K'))
    plt.xlabel('Cluster')
    plt.ylabel('Within-cluster Sum of Squares')
    

    #2 -- GAP
    plt.subplot(132)
    plt.title('Gap Statistics', size=15)
    plt.plot(number_clusters, gap, '-o')
    plt.ylabel('Gap')
    plt.xlabel('Cluster')
    plt.axvline(x=best_g, c='red')
    plt.legend(('GAP', 'Best K'))

    #3 -- SILHOUETTE
    plt.subplot(133)
    plt.title('Silhouette', size=15)
    plt.plot(number_clusters, sil, '-o')
    plt.ylabel('Silhouette score')
    plt.xlabel('Cluster')
    plt.axvline(x=best_s, c='red')
    plt.legend(('Silhouette', 'Best K'))

    plt.tight_layout()

    return plt

def plotCluster(new_plots, clusters):
    if len(new_plots) == 1: 
        fs = (6,6)
        ps = 110
    elif len(new_plots) == 2: 
        fs = (10,6)
        ps = 120
    elif len(new_plots) == 3: 
        fs = (15,4)
        ps = 130
    else:  
        fs = (15,8)
        ps = 230

    plt.figure(figsize=fs)

    for i, plot in enumerate(new_plots):

        position = ps + i + 1

        if len(plot['features']) == 2:
            plt.subplot(position)
            colors = plt.get_cmap('rainbow')(np.linspace(0.0, 1.0, len(set(clusters['cluster']))))
            outliers = False
            for clust_number in set(clusters['cluster']):
                # outliers in grey
                if clust_number == -1:
                    outliers = True
                    c=([0.4,0.4,0.4]) 
                else:
                    c = colors[clust_number]
                clust_set = clusters[clusters.cluster == clust_number]
                plt.scatter(clust_set[plot['features'][0]], clust_set[plot['features'][1]], color =c, s= 20, alpha = 0.85)      
            plt.title(plot['title'], size=15)
            plt.xlabel(plot['features'][0], size=13)
            plt.ylabel(plot['features'][1], size=13)

            if outliers:
                custom_lines = [Line2D([0], [0], marker='o', color=([0,0,0,0]), label='Outliers',
                          markerfacecolor=([0.4,0.4,0.4]), markersize=10)]
                plt.legend(custom_lines, ['Outliers'])

        if len(plot['features']) == 3:
            ax = plt.subplot(position, projection='3d')

            xs = clusters[plot['features'][0]]
            ys = clusters[plot['features'][1]]
            zs = clusters[plot['features'][2]]
            ax.scatter(xs, ys, zs, s=50, alpha=0.6, c=clusters['cluster'],cmap='rainbow')

            ax.set_xlabel(plot['features'][0])
            ax.set_ylabel(plot['features'][1])
            ax.set_zlabel(plot['features'][2])

            plt.title(plot['title'], size=15, pad=35)

    plt.tight_layout()

    return plt


# plot elbow, gap & silhouette
def plotAgglomerativeTrain(max_clusters, sil, best_s):
    number_clusters = range(1, max_clusters + 1)
    plt.figure()
    #1 -- SILHOUETTE
    plt.title('Silhouette', size=15)
    plt.plot(number_clusters, sil, '-o')
    plt.ylabel('Silhouette score')
    plt.xlabel('Cluster')
    plt.axvline(x=best_s, c='red')

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