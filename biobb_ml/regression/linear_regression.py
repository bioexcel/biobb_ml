#!/usr/bin/env python3

"""Module containing the LinearRegression class and the command line interface."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.regression.common import *


class LinearRegression():
    """Trains and tests a given dataset and calculates coefficients and predictions for a linear regression.
    Wrapper of the sklearn.linear_model.LinearRegression module
    Visit the 'sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linear%20regression#sklearn.linear_model.LinearRegression>'_. 

    Args:
        input_dataset_path (str): Path to the input dataset. Accepted formats: csv.
        output_results_path (str): Path to the output results file. Accepted formats: csv.
        output_test_table_path (str) (Optional): Path to the test table file. Accepted formats: csv.
        output_test_plot_path (str) (Optional): Path to the test plot file. Accepted formats: png.
        properties (dic):
            * **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
            * **scale** (*bool*) - (True) Whether the dataset should be scaled or not.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
            * **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_results_path, output_test_table_path=None, output_test_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_test_table_path": output_test_table_path, "output_test_plot_path": output_test_plot_path } 
        }

        # Properties specific for BB
        self.independent_vars = properties.get('independent_vars', [])
        self.target = properties.get('target', '')
        self.scale = properties.get('scale', True)
        self.predictions = properties.get('predictions', [])
        self.test_size = properties.get('test_size', 0.2)
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
        self.io_dict["out"]["output_results_path"] = check_output_path(self.io_dict["out"]["output_results_path"],"output_results_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_test_table_path"] = check_output_path(self.io_dict["out"]["output_test_table_path"],"output_test_table_path", True, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_test_plot_path"] = check_output_path(self.io_dict["out"]["output_test_plot_path"],"output_test_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the LinearRegression module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_results_path"],self.io_dict["out"]["output_test_table_path"],self.io_dict["out"]["output_test_plot_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])

        # declare inputs and targets
        targets = data[self.target]
        # the inputs are all the independent variables
        inputs = data.filter(self.independent_vars)

        t_inputs = inputs
        # scale dataset
        if self.scale:
            fu.log('Scaling dataset', out_log, self.global_log)
            scaler = StandardScaler()
            scaler.fit(t_inputs)
            t_inputs = scaler.transform(t_inputs)

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        x_train, x_test, y_train, y_test = train_test_split(t_inputs, targets, test_size=self.test_size, random_state=42)

        # regression
        fu.log('Training dataset applying linear regression', out_log, self.global_log)
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)

        # scores and coefficients
        score = reg.score(x_train, y_train)
        bias = reg.intercept_
        coef = reg.coef_
        coef = [ '%.3f' % item for item in coef ]
        adj_r2 = adjusted_r2(x_train, y_train, score)
        p_values = f_regression(x_train, y_train)[1]
        p_values = [ '%.3f' % item for item in p_values ]

        # r-squared
        r2_table = pd.DataFrame()
        r2_table["feature"] = ['R-squared','Adj. R-squared']
        r2_table['coefficient'] = [score, adj_r2]

        # p-values
        cols = ['bias']
        cols.extend(self.independent_vars)
        coefs_table = pd.DataFrame(cols, columns=['feature'])
        c = [round(bias, 3)]
        c.extend(coef)
        c = list(map(float, c))
        coefs_table['coefficient'] = c
        p = [0]
        p.extend(p_values)
        
        coefs_table['p-value'] = p
        fu.log('Calculating scores and coefficients for training dataset\n\nR2 AND ADJUSTED R2\n\n%s\n\nP-VALUES\n\n%s\n' % (r2_table, coefs_table), out_log, self.global_log)

        # testing
        # predict data from x_test
        y_hat_test = reg.predict(x_test)
        test_table = pd.DataFrame(y_hat_test, columns=['prediction'])
        # reset y_test (problem with old indexes column)
        y_test = y_test.reset_index(drop=True)
        # add real values to predicted ones in test_table table
        test_table['target'] = y_test
        # calculate difference between target and prediction (absolute and %)
        test_table['residual'] = test_table['target'] - test_table['prediction']
        test_table['difference %'] = np.absolute(test_table['residual']/test_table['target']*100)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        # sort by difference in %
        test_table = test_table.sort_values(by=['difference %'])
        test_table = test_table.reset_index(drop=True)
        fu.log('Testing\n\nTEST DATA\n\n%s\n' % test_table, out_log, self.global_log)

        coeffs_test = 'Residual sum of squares: %.2f\nVariance score: %.2f' % (np.mean((y_hat_test - y_test) ** 2), reg.score(x_test, y_test))
        fu.log('Calculating scores and coefficients for testing dataset\n\nCOEFFICIENTS\n\n%s\n' % coeffs_test, out_log, self.global_log)

        if(self.io_dict["out"]["output_test_table_path"]): 
            fu.log('Saving testing data to %s' % self.io_dict["out"]["output_test_table_path"], out_log, self.global_log)
            test_table.to_csv(self.io_dict["out"]["output_test_table_path"], index = False, header=True)

        # create test plot
        if(self.io_dict["out"]["output_test_plot_path"]): 
            fu.log('Saving testing plot to %s' % self.io_dict["out"]["output_test_plot_path"], out_log, self.global_log)
            plt.scatter(y_test, y_hat_test, alpha=0.2)
            plt.xlabel('targets',size=18)
            plt.ylabel('predictions',size=18)
            plt.savefig(self.io_dict["out"]["output_test_plot_path"], dpi=150)

        # prediction
        new_data_table = pd.DataFrame(data=get_list_of_predictors(self.predictions),columns=self.independent_vars)
        new_data = new_data_table
        if self.scale:
            new_data = scaler.transform(new_data_table)
        p = reg.predict(new_data)
        p = np.around(p, 2)
        new_data_table[self.target] = p
        fu.log('Predicting results\n\nPREDICTION RESULTS\n\n%s\n' % new_data_table, out_log, self.global_log)
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        new_data_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and calculates coefficients and predictions for a linear regression.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--output_test_table_path', required=False, help='Path to the test table file. Accepted formats: csv.')
    parser.add_argument('--output_test_plot_path', required=False, help='Path to the test plot file. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    LinearRegression(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_test_table_path=args.output_test_table_path, 
                   output_test_plot_path=args.output_test_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

