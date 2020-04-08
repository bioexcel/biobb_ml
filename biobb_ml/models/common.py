""" Common functions for package biobb_analysis.ambertools """
from pathlib import Path, PurePath
import shutil
from biobb_common.tools import file_utils as fu

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