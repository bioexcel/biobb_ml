""" Common functions for package biobb_analysis.ambertools """
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from random import sample
from math import isnan
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
		'output_wcss_path': ['csv'],
		'output_gap_path': ['csv'],
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

def getSilhouetthe(method, X, max_clusters):
    # Run clustering with different k and check the metrics
    silhouette_list = []

    k_list = list(range(2,max_clusters + 1))
    for p in k_list:

        if method == 'kmeans': clusterer = KMeans(p)
        elif method == 'agglomerative': clusterer = AgglomerativeClustering(n_clusters=p, linkage="average")

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
    plt.xlabel('Cluster')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.axvline(x=best_k, c='red')

    #2 -- GAP
    plt.subplot(132)
    plt.title('Gap Statistics', size=15)
    plt.plot(number_clusters, gap, '-o')
    plt.ylabel('Gap')
    plt.xlabel('Cluster')
    plt.axvline(x=best_g, c='red')

    #3 -- SILHOUETTE
    plt.subplot(133)
    plt.title('Silhouette', size=15)
    plt.plot(number_clusters, sil, '-o')
    plt.ylabel('Silhouette score')
    plt.xlabel('Cluster')
    plt.axvline(x=best_s, c='red')

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
            plt.scatter(clusters[plot['features'][0]],clusters[plot['features'][1]],c=clusters['cluster'],cmap='rainbow')
            plt.title(plot['title'], size=15)
            plt.xlabel(plot['features'][0], size=13)
            plt.ylabel(plot['features'][1], size=13)

        if len(plot['features']) == 3:
            ax = plt.subplot(position, projection='3d')

            xs = clusters[plot['features'][0]]
            ys = clusters[plot['features'][1]]
            zs = clusters[plot['features'][2]]
            ax.scatter(xs, ys, zs, s=50, alpha=0.6, c=clusters['cluster'],cmap='rainbow')

            ax.set_xlabel(plot['features'][0])
            ax.set_ylabel(plot['features'][1])
            ax.set_zlabel(plot['features'][2])

            plt.title(plot['title'], size=15)

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