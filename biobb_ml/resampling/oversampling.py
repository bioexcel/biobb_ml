#!/usr/bin/env python3

"""Module containing the Oversampling class and the command line interface."""
import argparse
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


class Oversampling():
    """
    | biobb_ml Oversampling
    | Wrapper of most of the imblearn.over_sampling methods.
    | Involves supplementing the training data with multiple copies of some of the minority classes of a given dataset. If regression is specified as type, the data will be resampled to classes in order to apply the oversampling model. Visit the imbalanced-learn official website for the different methods accepted in this wrapper: `RandomOverSampler <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html>`_, `SMOTE <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html>`_, `BorderlineSMOTE <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.BorderlineSMOTE.html>`_, `SVMSMOTE <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SVMSMOTE.html>`_, `ADASYN <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html>`_

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/resampling/dataset_resampling.csv>`_. Accepted formats: csv (edam:format_3752).
        output_dataset_path (str): Path to the output dataset. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/resampling/ref_output_oversampling.csv>`_. Accepted formats: csv (edam:format_3752).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **method** (*str*) - (None) Oversampling method. It's a mandatory property. Values: random (`RandomOverSampler <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html>`_: Object to over-sample the minority classes by picking samples at random with replacement), smote (`SMOTE <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html>`_: This object is an implementation of SMOTE - Synthetic Minority Over-sampling Technique), borderline (`BorderlineSMOTE <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.BorderlineSMOTE.html>`_: This algorithm is a variant of the original SMOTE algorithm. Borderline samples will be detected and used to generate new synthetic samples), svmsmote (`SVMSMOTE <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SVMSMOTE.html>`_: Variant of SMOTE algorithm which use an SVM algorithm to detect sample to use for generating new synthetic samples), adasyn (`ADASYN <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html>`_: Perform over-sampling using Adaptive Synthetic -ADASYN- sampling approach for imbalanced datasets).
            * **type** (*str*) - (None) Type of oversampling. It's a mandatory property. Values: regression (the oversampling will be applied on a continuous dataset), classification (the oversampling will be applied on a classified dataset).
            * **target** (*dict*) - ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **evaluate** (*bool*) - (False)  Whether or not to evaluate the dataset before and after applying the resampling.
            * **evaluate_splits** (*int*) - (3) [2~100|1] Number of folds to be applied by the Repeated Stratified K-Fold evaluation method. Must be at least 2.
            * **evaluate_repeats** (*int*) - (3) [2~100|1] Number of times Repeated Stratified K-Fold cross validator needs to be repeated.
            * **n_bins** (*int*) - (5) [1~100|1] Only for regression oversampling. The number of classes that the user wants to generate with the target data.
            * **balanced_binning** (*bool*) - (False)  Only for regression oversampling. Decides whether samples are to be distributed roughly equally across all classes.
            * **sampling_strategy** (*dict*) - ({ "target": "auto" })  Sampling information to sample the data set. Formats: { "target": "auto" }, { "ratio": 0.3 }, { "dict": { 0: 300, 1: 200, 2: 100 } } or { "list": [0, 2, 3] }. When "target", specify the class targeted by the resampling; the number of samples in the different classes will be equalized; possible choices are: minority (resample only the minority class), not minority (resample all classes but the minority class), not majority (resample all classes but the majority class), all (resample all classes), auto (equivalent to 'not majority'). When "ratio", it corresponds to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling (ONLY IN CASE OF BINARY CLASSIFICATION).  When "dict", the keys correspond to the targeted classes, the values correspond to the desired number of samples for each targeted class. When "list", the list contains the classes targeted by the resampling.
            * **k_neighbors** (*int*) - (5) [1~100|1] Only for SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN. The number of nearest neighbours used to construct synthetic samples.
            * **random_state_method** (*int*) - (5) [1~1000|1] Controls the randomization of the algorithm.
            * **random_state_evaluate** (*int*) - (5) [1~1000|1] Controls the shuffling applied to the Repeated Stratified K-Fold evaluation method.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.resampling.oversampling import oversampling
            prop = { 
                'method': 'random, 
                'type': 'regression, 
                'target': { 
                    'column': 'target' 
                }, 
                'evaluate': true, 
                'n_bins': 10,
                'sampling_strategy': { 
                    'target': 'minority' 
                }
            }
            oversampling(input_dataset_path='/path/to/myDataset.csv', 
                        output_dataset_path='/path/to/newDataset.csv', 
                        properties=prop)

    Info:
        * wrapped_software:
            * name: imbalanced-learn over_sampling
            * version: >0.7.0
            * license: MIT
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_dataset_path, output_dataset_path, 
                properties=None, **kwargs) -> None:
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
        self.evaluate_splits = properties.get('evaluate_splits', 3)
        self.evaluate_repeats = properties.get('evaluate_repeats', 3)
        self.n_bins = properties.get('n_bins', 5)
        self.balanced_binning = properties.get('balanced_binning', False)
        self.sampling_strategy = properties.get('sampling_strategy', { 'target': 'auto' })
        self.k_neighbors = properties.get('k_neighbors', 5)
        self.random_state_method = properties.get('random_state_method', 5)
        self.random_state_evaluate = properties.get('random_state_evaluate', 5)
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
        """Execute the :class:`Oversampling <resampling.oversampling.Oversampling>` resampling.oversampling.Oversampling object."""

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
        method = getResamplingMethod(self.method, 'oversampling', out_log, self.__class__.__name__)
        checkResamplingType(self.type, out_log, self.__class__.__name__)
        sampling_strategy = getSamplingStrategy(self.sampling_strategy, out_log, self.__class__.__name__)

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        if 'column' in self.target:
            labels = getHeader(self.io_dict["in"]["input_dataset_path"])
            skiprows = 1
            header = 0
        else:
            labels = None
            skiprows = None
            header = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = None, sep="\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)

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
        # calling oversample method
        if self.method == 'random':
            method = method(sampling_strategy=sampling_strategy, random_state=self.random_state_method)
        elif self.method == 'smote':
            method = method(sampling_strategy=sampling_strategy, k_neighbors=self.k_neighbors, random_state=self.random_state_method)
        elif self.method == 'borderline':
            method = method(sampling_strategy=sampling_strategy, k_neighbors=self.k_neighbors, random_state=self.random_state_method)
        elif self.method == 'svmsmote':
            method = method(sampling_strategy=sampling_strategy, k_neighbors=self.k_neighbors, random_state=self.random_state_method)
        elif self.method == 'adasyn':
            method = method(sampling_strategy=sampling_strategy, n_neighbors=self.k_neighbors, random_state=self.random_state_method)
        
        fu.log('Target: %s' % (getTargetValue(self.target, out_log, self.__class__.__name__)), out_log, self.global_log)

        # oversampling
        if self.type == 'regression':
            fu.log('Oversampling regression dataset, continuous data will be classified', out_log, self.global_log)
            # call resampler class for Regression ReSampling            
            rs = resampler()
            # Create n_bins classes for the dataset
            ranges, y, target_pos = rs.fit(train_df, target=getTargetValue(self.target, out_log, self.__class__.__name__), bins=self.n_bins, balanced_binning=self.balanced_binning, verbose=0)
            # Get the over-sampled data
            final_X, final_y = rs.resample(method, train_df, y)
        elif self.type == 'classification':
            # get X and y
            y = getTarget(self.target, train_df, out_log, self.__class__.__name__)
            # fit and resample
            final_X, final_y = method.fit_resample(X, y)
            target_pos = None

        # evaluate oversampling
        if self.evaluate:
            fu.log('Evaluating data before oversampling with RandomForestClassifier', out_log, self.global_log)
            cv = RepeatedStratifiedKFold(n_splits=self.evaluate_splits, n_repeats=self.evaluate_repeats, random_state=self.random_state_evaluate)
            # evaluate model
            scores = cross_val_score(RandomForestClassifier(), X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            if not np.isnan(np.mean(scores)):
                fu.log('Mean Accuracy before oversampling: %.3f' % (np.mean(scores)), out_log, self.global_log)
            else:
                fu.log('Unable to calculate cross validation score, NaN was returned.', out_log, self.global_log)

        # log distribution before oversampling
        dist = ''
        for k,v in Counter(y).items():
            per = v / len(y) * 100
            rng = ''
            if ranges: rng = str(ranges[k])
            dist = dist + 'Class=%d, n=%d (%.3f%%) %s\n' % (k, v, per, rng)
        fu.log('Classes distribution before oversampling:\n\n%s' % dist, out_log, self.global_log)

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

        # if no header, target is in a different column
        if target_pos: t = target_pos
        else: t = getTargetValue(self.target, out_log, self.__class__.__name__)
        # log distribution after oversampling
        if self.type == 'regression':
            ranges, y_out, _ = rs.fit(out_df, target=t, bins=self.n_bins, balanced_binning=self.balanced_binning, verbose=0)
        elif self.type == 'classification':
            y_out = getTarget(self.target, out_df, out_log, self.__class__.__name__)

        dist = ''
        for k,v in Counter(y_out).items():
            per = v / len(y_out) * 100
            rng = ''
            if ranges: rng = str(ranges[k])
            dist = dist + 'Class=%d, n=%d (%.3f%%) %s\n' % (k, v, per, rng)
        fu.log('Classes distribution after oversampling:\n\n%s' % dist, out_log, self.global_log)

        # evaluate oversampling
        if self.evaluate:
            fu.log('Evaluating data after oversampling with RandomForestClassifier', out_log, self.global_log)
            cv = RepeatedStratifiedKFold(n_splits=self.evaluate_splits, n_repeats=self.evaluate_repeats, random_state=self.random_state_evaluate)
            # evaluate model
            scores = cross_val_score(RandomForestClassifier(), final_X, y_out, scoring='accuracy', cv=cv, n_jobs=-1)
            if not np.isnan(np.mean(scores)):
                fu.log('Mean Accuracy after oversampling a %s dataset with %s method: %.3f' % (self.type, oversampling_methods[self.method]['method'], np.mean(scores)), out_log, self.global_log)
            else:
                fu.log('Unable to calculate cross validation score, NaN was returned.', out_log, self.global_log)

        # save output
        hdr = False
        if header == 0: hdr = True
        fu.log('Saving oversampled dataset to %s' % self.io_dict["out"]["output_dataset_path"], out_log, self.global_log)
        out_df.to_csv(self.io_dict["out"]["output_dataset_path"], index = False, header=hdr)

        return 0

def oversampling(input_dataset_path: str, output_dataset_path: str, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`Oversampling <resampling.oversampling.Oversampling>` class and
    execute the :meth:`launch() <resampling.oversampling.Oversampling.launch>` method."""

    return Oversampling(input_dataset_path=input_dataset_path,
                   output_dataset_path=output_dataset_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of most of the imblearn.over_sampling methods.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_dataset_path', required=True, help='Path to the output dataset. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    oversampling(input_dataset_path=args.input_dataset_path,
                   output_dataset_path=args.output_dataset_path,
                   properties=properties)

if __name__ == '__main__':
    main()

