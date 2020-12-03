#!/usr/bin/env python3

"""Module containing the Undersampling class and the command line interface."""
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from biobb_ml.resampling.reg_resampler import resampler
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.resampling.common import *


class Undersampling():
    """
    | biobb_ml Undersampling
    | Wrapper of most of the imblearn.under_sampling methods
    | Remove samples from the majority class of a given dataset, with or without replacement. If regression is specified as type, the data will be resampled to classes in order to apply the undersampling model. Visit the imbalanced-learn official website for the different methods accepted in this wrapper: `RandomUnderSampler <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html>`_, `NearMiss <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html>`_, `CondensedNearestNeighbour <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.CondensedNearestNeighbour.html>`_, `TomekLinks <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.TomekLinks.html>`_, `EditedNearestNeighbours <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.EditedNearestNeighbours.html>`_, `NeighbourhoodCleaningRule <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NeighbourhoodCleaningRule.html>`_, `ClusterCentroids <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.ClusterCentroids.html>`_.  

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <>`_. Accepted formats: csv.
        output_dataset_path (str): Path to the output dataset. File type: output. `Sample file <>`_. Accepted formats: csv.
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **method** (*str*) - (None) Undersampling method. It's a mandatory property. Values: random (`RandomUnderSampler <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html>`_), nearmiss (`NearMiss <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html>`_), cnn (`CondensedNearestNeighbour <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.CondensedNearestNeighbour.html>`_), tomeklinks (`TomekLinks <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.TomekLinks.html>`_), enn (`EditedNearestNeighbours <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.EditedNearestNeighbours.html>`_), ncr (`NeighbourhoodCleaningRule <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NeighbourhoodCleaningRule.html>`_), cluster (`ClusterCentroids <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.ClusterCentroids.html>`_).
            * **type** (*str*) - (None) Type of undersampling. It's a mandatory property. Values: regression (dataset is for a regression analysis), classification (dataset is for a classification analysis).
            * **target** (*dict*) - ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **evaluate** (*bool*) - (False)  Whether or not to evaluate the dataset before and after applying the resampling.
            * **n_bins** (*int*) - (5) Only for regression undersampling. The number of classes that the user wants to generate with the target data.
            * **balanced_binning** (*bool*) - (False)  Only for regression undersampling. Decides whether samples are to be distributed roughly equally across all classes.
            * **sampling_strategy** (*dict*) - ({ "target": "auto" })  Sampling information to sample the data set. Formats: { "target": "auto" }, { "ratio": 0.3 }, { "dict": { 0: 300, 1: 200, 2: 100 } } or { "list": [0, 2, 3] }. When "target", specify the class targeted by the resampling; the number of samples in the different classes will be equalized; possible choices are: majority (resample only the majority class), not minority (resample all classes but the minority class), not majority (resample all classes but the majority class), all (resample all classes), auto (equivalent to 'not minority'). When "ratio", it corresponds to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling (ONLY IN CASE OF BINARY CLASSIFICATION). When "dict", the keys correspond to the targeted classes, the values correspond to the desired number of samples for each targeted class. When "list", the list contains the classes targeted by the resampling.
            * **version** (*int*) - (1) Only for NearMiss method. Version of the NearMiss to use. Values: 1, 2, 3.
            * **n_neighbors** (*int*) - (1) Only for NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours and NeighbourhoodCleaningRule methods. Size of the neighbourhood to consider to compute the average distance to the minority point samples.
            * **threshold_cleaning** (*float*) - (0.5) Only for NeighbourhoodCleaningRule method. Threshold used to whether consider a class or not during the cleaning after applying ENN.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Info:
        * wrapped_software:
            * name: imbalanced-learn.under_sampling
            * version: >0.7.0
            * license: MIT
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_dataset_path, 
                 output_dataset_path, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_dataset_path": output_dataset_path } 
        }

        # Properties specific for BB
        self.method = properties.get('method', None)
        self.type = properties.get('type', None)
        self.target = properties.get('target', {})
        self.evaluate = properties.get('evaluate', False)
        self.n_bins = properties.get('n_bins', 5)
        self.balanced_binning = properties.get('balanced_binning', False)
        self.sampling_strategy = properties.get('sampling_strategy', { 'target': 'auto' })
        self.version = properties.get('version', 1)
        self.n_neighbors = properties.get('n_neighbors', 1)
        self.threshold_cleaning = properties.get('threshold_cleaning', 1)
        self.properties = properties

        # Properties common in all BB
        self.can_write_console_log = properties.get('can_write_console_log', True)
        self.global_log = properties.get('global_log', None)
        self.prefix = properties.get('prefix', None)
        self.step = properties.get('step', None)
        self.path = properties.get('path', '')
        self.remove_tmp = properties.get('remove_tmp', True)
        self.restart = properties.get('restart', False)

    def check_data_params(self, out_log, err_log):
        """ Checks all the input/output paths and parameters """
        self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", out_log, self.__class__.__name__)
        self.io_dict["out"]["output_dataset_path"] = check_output_path(self.io_dict["out"]["output_dataset_path"],"output_dataset_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the Undersampling module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_dataset_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # check mandatory properties
        method = getResamplingMethod(self.method, 'undersampling', out_log, self.__class__.__name__)
        checkResamplingType(self.type, out_log, self.__class__.__name__)
        sampling_strategy = getSamplingStrategy(self.sampling_strategy, out_log, self.__class__.__name__)

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        if 'column' in self.target:
            header = 0
        else:
            header = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = header, sep="\s+|;|:|,|\t", engine="python")

        train_df = data
        ranges = None

        le = preprocessing.LabelEncoder()

        cols_encoded = []
        for column in train_df:
            # if type object, LabelEncoder.fit_transform
            if train_df[column].dtypes == 'object':
                cols_encoded.append(column)
                train_df[column] = le.fit_transform(train_df[column])

        # defining X
        X = train_df.loc[:, train_df.columns != getTargetValue(self.target, out_log, self.__class__.__name__)] 
        # calling undersample method
        if self.method == 'random':
            method = method(sampling_strategy=sampling_strategy)
        elif self.method == 'nearmiss':
            if self.version == 3:
                method = method(sampling_strategy=sampling_strategy, version=self.version, n_neighbors_ver3=self.n_neighbors)
            else: 
                method = method(sampling_strategy=sampling_strategy, version=self.version, n_neighbors=self.n_neighbors)
        elif self.method == 'cnn':
            method = method(sampling_strategy=sampling_strategy, n_neighbors=self.n_neighbors)
        elif self.method == 'tomeklinks':
            method = method(sampling_strategy=sampling_strategy)
        elif self.method == 'enn':
            method = method(sampling_strategy=sampling_strategy, n_neighbors=self.n_neighbors)
        elif self.method == 'ncr':
            method = method(sampling_strategy=sampling_strategy, n_neighbors=self.n_neighbors, threshold_cleaning=self.threshold_cleaning)
        elif self.method == 'cluster':
            method = method(sampling_strategy=sampling_strategy)

        # undersampling
        if self.type == 'regression':
            fu.log('Undersampling regression dataset, continuous data will be classified', out_log, self.global_log)
            # call resampler class for Regression ReSampling            
            rs = resampler()
            # Create n_bins classes for the dataset
            ranges, y = rs.fit(train_df, target=getTargetValue(self.target, out_log, self.__class__.__name__), bins=self.n_bins, balanced_binning=self.balanced_binning, verbose=0)
            # Get the under-sampled data
            final_X, final_y = rs.resample(method, train_df, y)
        elif self.type == 'classification':
            # get X and y
            y = getTarget(self.target, train_df, out_log, self.__class__.__name__)
            # fit and resample
            final_X, final_y = method.fit_resample(X, y)

        # evaluate undersampling
        if self.evaluate:
            fu.log('Evaluating data before undersampling with RandomForestClassifier', out_log, self.global_log)
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
            # evaluate model
            scores = cross_val_score(RandomForestClassifier(class_weight='balanced'), X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            if not np.isnan(np.mean(scores)):
                fu.log('Mean Accuracy before undersampling: %.3f' % (np.mean(scores)), out_log, self.global_log)
            else:
                fu.log('Unable to calculate cross validation score, NaN was returned.', out_log, self.global_log)
        
        # log distribution before undersampling
        dist = ''        
        for k,v in Counter(y).items():
            per = v / len(y) * 100
            rng = ''
            if ranges: rng = str(ranges[k])
            dist = dist + 'Class=%d, n=%d (%.3f%%) %s\n' % (k, v, per, rng)
        fu.log('Classes distribution before undersampling:\n\n%s' % dist, out_log, self.global_log)

        # join final_X and final_y in the output dataframe
        if header is None:
            # numpy
            out_df = np.column_stack((final_X, final_y))
        else:
            # pandas
            out_df = final_X.join(final_y)
                
        # if no header, convert np to pd
        if header is None: out_df = pd.DataFrame(data=out_df)

        # if cols encoded, decode them
        if cols_encoded:
            for column in cols_encoded:
                if header is None:
                    out_df = out_df.astype({column: int } ) 
                out_df[column] = le.inverse_transform(out_df[column].values.ravel())

        # log distribution after undersampling
        if self.type == 'regression':
            ranges, y_out = rs.fit(out_df, target=getTargetValue(self.target, out_log, self.__class__.__name__), bins=self.n_bins, balanced_binning=self.balanced_binning, verbose=0)
        elif self.type == 'classification':
            y_out = getTarget(self.target, out_df, out_log, self.__class__.__name__)

        dist = ''
        for k,v in Counter(y_out).items():
            per = v / len(y_out) * 100
            rng = ''
            if ranges: rng = str(ranges[k])
            dist = dist + 'Class=%d, n=%d (%.3f%%) %s\n' % (k, v, per, rng)
        fu.log('Classes distribution after undersampling:\n\n%s' % dist, out_log, self.global_log)

        # evaluate undersampling
        if self.evaluate:
            fu.log('Evaluating data after undersampling with RandomForestClassifier', out_log, self.global_log)
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
            # evaluate model
            scores = cross_val_score(RandomForestClassifier(class_weight='balanced'), final_X, y_out, scoring='accuracy', cv=cv, n_jobs=-1)
            if not np.isnan(np.mean(scores)):
                fu.log('Mean Accuracy after undersampling a %s dataset with %s method: %.3f' % (self.type, undersampling_methods[self.method]['method'], np.mean(scores)), out_log, self.global_log)
            else:
                fu.log('Unable to calculate cross validation score, NaN was returned.', out_log, self.global_log)

        # save output
        hdr = False
        if header == 0: hdr = True
        fu.log('Saving undersampled dataset to %s' % self.io_dict["out"]["output_dataset_path"], out_log, self.global_log)
        out_df.to_csv(self.io_dict["out"]["output_dataset_path"], index = False, header=hdr)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Remove samples from the majority class of a given dataset, with or without replacement. If regression is specified as type, the data will be resampled to classes in order to apply the undersampling model.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_dataset_path', required=True, help='Path to the output dataset. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    Undersampling(input_dataset_path=args.input_dataset_path,
                   output_dataset_path=args.output_dataset_path,
                   properties=properties).launch()

if __name__ == '__main__':
    main()

