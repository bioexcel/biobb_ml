#!/usr/bin/env python3

"""Module containing the Cpptraj Average class and the command line interface."""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.models.common import *


class LinearRegression():
    """Trains and tests a given dataset and calculates coefficients and predictions for a linear regression.
    Wrapper of the sklearn.linear_model.LinearRegression module
    Visit the 'sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linear%20regression#sklearn.linear_model.LinearRegression>'_. 

    Args:
        input_dataset_path (str): Path to the input dataset. Accepted formats: csv.
        output_results_path (str): Path to the output results file. Accepted formats: csv.
        output_coefs_path (str): Path to the output coefficients file. Accepted formats: csv.
        properties (dic):
            * **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **predictions** (*dict*) - (None) Dictionary with all values you want to predict targets.
            * **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
            * **random_state** (*int*) - (42) The seed used by the random number generator.
            * **shuffle** (*bool*) - (True) Whether or not to shuffle the data before splitting.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_results_path, output_coefs_path, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_coefs_path": output_coefs_path } 
        }

        # Properties specific for BB
        self.independent_vars = properties.get('independent_vars', [])
        self.target = properties.get('target', '')
        self.predictions = properties.get('predictions', [])
        self.test_size = properties.get('test_size', 0.2)
        self.random_state = properties.get('random_state', 42)
        self.shuffle = properties.get('shuffle', True)
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
        # TODO check params
        """self.io_dict["in"]["input_top_path"], self.input_top_path_orig = check_top_path(self.io_dict["in"]["input_top_path"], out_log, self.__class__.__name__)
        self.io_dict["in"]["input_traj_path"] = check_traj_path(self.io_dict["in"]["input_traj_path"], out_log, self.__class__.__name__)
        self.io_dict["out"]["output_cpptraj_path"] = check_out_path(self.io_dict["out"]["output_cpptraj_path"], out_log, self.__class__.__name__)
        self.in_parameters = get_parameters(self.properties, 'in_parameters', self.__class__.__name__, out_log)
        self.out_parameters = get_parameters(self.properties, 'out_parameters', self.__class__.__name__, out_log)"""

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the Ambertools cpptraj module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_cpptraj_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])
        
        # declare inputs and targets
        targets = data[self.target]
        inputs = data.drop([self.target],axis=1)

        # scale dataset
        scaler = StandardScaler()
        scaler.fit(inputs)
        inputs_scaled = scaler.transform(inputs)

        # train / test split
        x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=self.test_size, random_state=self.random_state)

        # regression
        reg = linear_model.LinearRegression()
        reg.fit(x_train,y_train)

        # scores and coefficients
        score = reg.score(x_train,y_train)
        bias = reg.intercept_
        coef = reg.coef_
        adj_r2 = adjusted_r2(x_train,y_train, score)
        #p_values = f_regression(x_train,y_train)[1]

        # coefs table
        reg_summary = pd.DataFrame([['r-squared'],['adj. r-sq'],['bias'],['size'],['year']], columns=['feature'])
        reg_summary['coefficient'] = score, adj_r2, bias, coef[0], coef[1]
        #reg_summary['p-values'] = '-', '-', '-', p_values[0].round(3), p_values[1].round(3)
        print(reg_summary)

        # TODO: create difference table train/test and save somewhere. See S35_L233

        # prediction
        new_data = pd.DataFrame(data=get_list_of_predictors(self.predictions),columns=self.independent_vars)
        new_data_scaled = scaler.transform(new_data)
        p = reg.predict(new_data_scaled)
        p = np.around(p, 2)
        new_data[self.target] = p
        print(new_data)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and calculates coefficients and predictions for a linear regression.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--output_coefs_path', required=False, help='Path to the output coefficients file. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    LinearRegression(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_coefs_path=args.output_coefs_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

