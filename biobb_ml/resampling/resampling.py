#!/usr/bin/env python3

"""Module containing the Resampling class and the command line interface."""
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


class Resampling():
    """Combine over- and under-sampling methods to remove samples and supplement the dataset. If regression is specified as type, the data will be resampled to classes in order to apply the resampling model.
    Wrapper of the imblearn.combine methods
    Visit the imbalanced-learn official website for the different methods accepted in this wrapper: 
    `SMOTETomek <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTETomek.html>`_
    `SMOTEENN <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTEENN.html>`_
    

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <>`_. Accepted formats: csv.
        output_dataset_path (str): Path to the output dataset. File type: output. `Sample file <>`_. Accepted formats: csv.
        properties (dic):
            * **method** (*str*) - (None) Resampling method. It's a mandatory property. Values: smotetomek (`SMOTETomek <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTETomek.html>`_), smotenn (`SMOTEENN <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.combine.SMOTEENN.html>`_).
            * **type** (*str*) - (None) Type of resampling. It's a mandatory property. Values: regression, classification.
            * **target** (*dict*) - (None) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **evaluate** (*bool*) - (False)  Whether or not to evaluate the dataset befaore and after applying the resampling.
            * **n_bins** (*int*) - (5) Only for regression resampling. The number of classes that the user wants to generate with the target data.
            * **balanced_binning** (*bool*) - (False)  Only for regression resampling. Decides whether samples are to be distributed roughly equally across all classes.
            * **sampling_strategy** (*str*) - ("auto")  Sampling information to sample the data set. ONLY IN CASE OF BINARY CLASSIFICATION: A float corresponding to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling can be passed. Values: minority (resample only the minority class), not minority (resample all classes but the minority class), not majority (resample all classes but the majority class), all (resample all classes), auto (equivalent to 'not majority').
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
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
        self.target = properties.get('target', None)
        self.evaluate = properties.get('evaluate', False)
        self.n_bins = properties.get('n_bins', 5)
        self.balanced_binning = properties.get('balanced_binning', False)
        self.sampling_strategy = properties.get('sampling_strategy', 'auto')
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
        """Launches the execution of the Resampling module."""

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

        method = getResamplingMethod(self.method, 'resampling', out_log, self.__class__.__name__)
        checkResamplingType(self.type, out_log, self.__class__.__name__)

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        if 'column' in self.target:
            header = 0
        else:
            header = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = header, sep="\s+|;|:|,|\t", engine="python")

        train_df = data

        le = preprocessing.LabelEncoder()

        cols_encoded = []
        for column in train_df:
            # if type object, LabelEncoder.fit_transform
            if train_df[column].dtypes == 'object':
                cols_encoded.append(column)
                train_df[column] = le.fit_transform(train_df[column])

        # defining X
        X = train_df.loc[:, train_df.columns != getTargetValue(self.target, out_log, self.__class__.__name__)] 
        # calling resample method
        if self.method == 'smotetomek':
            method = method(sampling_strategy=self.sampling_strategy)
        elif self.method == 'smotenn':
            method = method(sampling_strategy=self.sampling_strategy)

        # resampling
        if self.type == 'regression':
            fu.log('Resampling regression dataset, continuous data will be classified', out_log, self.global_log)
            # call resampler class for Regression ReSampling            
            rs = resampler()
            # Create n_bins classes for the dataset
            y = rs.fit(train_df, target=getTargetValue(self.target, out_log, self.__class__.__name__), bins=self.n_bins, balanced_binning=self.balanced_binning, verbose=0)
            # Get the re-sampled data
            final_X, final_y = rs.resample(method, train_df, y)
        elif self.type == 'classification':
            # get X and y
            y = getTarget(self.target, train_df, out_log, self.__class__.__name__)
            # fit and resample
            final_X, final_y = method.fit_resample(X, y)

        # evaluate resampling
        if self.evaluate:
            fu.log('Evaluating data before resampling with RandomForestClassifier', out_log, self.global_log)
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
            # evaluate model
            scores = cross_val_score(RandomForestClassifier(class_weight='balanced'), X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            if not np.isnan(np.mean(scores)):
                fu.log('Mean Accuracy before resampling: %.3f' % (np.mean(scores)), out_log, self.global_log)
            else:
                fu.log('Unable to calculate cross validation score, NaN was returned.', out_log, self.global_log)
        
        # log distribution before resampling
        dist = ''
        for k,v in Counter(y).items():
            per = v / len(y) * 100
            dist = dist + 'Class=%d, n=%d (%.3f%%)\n' % (k, v, per)
        fu.log('Classes distribution before resampling:\n\n%s' % dist, out_log, self.global_log)

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

        # log distribution after resampling
        if self.type == 'regression':
            y_out = rs.fit(out_df, target=getTargetValue(self.target, out_log, self.__class__.__name__), bins=self.n_bins, balanced_binning=self.balanced_binning, verbose=0)
        elif self.type == 'classification':
            y_out = getTarget(self.target, out_df, out_log, self.__class__.__name__)

        dist = ''
        for k,v in Counter(y_out).items():
            per = v / len(y_out) * 100
            dist = dist + 'Class=%d, n=%d (%.3f%%)\n' % (k, v, per)
        fu.log('Classes distribution after resampling:\n\n%s' % dist, out_log, self.global_log)

        # evaluate resampling
        if self.evaluate:
            fu.log('Evaluating data after resampling with RandomForestClassifier', out_log, self.global_log)
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)
            # evaluate model
            scores = cross_val_score(RandomForestClassifier(class_weight='balanced'), final_X, y_out, scoring='accuracy', cv=cv, n_jobs=-1)
            if not np.isnan(np.mean(scores)):
                fu.log('Mean Accuracy after resampling a %s dataset with %s method: %.3f' % (self.type, resampling_methods[self.method]['method'], np.mean(scores)), out_log, self.global_log)
            else:
                fu.log('Unable to calculate cross validation score, NaN was returned.', out_log, self.global_log)

        # save output
        hdr = False
        if header == 0: hdr = True
        fu.log('Saving resampled dataset to %s' % self.io_dict["out"]["output_dataset_path"], out_log, self.global_log)
        out_df.to_csv(self.io_dict["out"]["output_dataset_path"], index = False, header=hdr)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Combine over- and under-sampling methods to remove samples and supplement the dataset. If regression is specified as type, the data will be resampled to classes in order to apply the resampling model.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_dataset_path', required=True, help='Path to the output dataset. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    Resampling(input_dataset_path=args.input_dataset_path,
                   output_dataset_path=args.output_dataset_path,
                   properties=properties).launch()

if __name__ == '__main__':
    main()

