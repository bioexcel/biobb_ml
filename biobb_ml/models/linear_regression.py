#!/usr/bin/env python3

"""Module containing the LinearRegression class and the command line interface."""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
        output_percent_path (str): Path to the output percentages table file. Accepted formats: csv.
        properties (dic):
            * **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
            * **scale** (*bool*) - (True) Whether the dataset should be scaled or not.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
            * **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
            * **random_state** (*int*) - (42) The seed used by the random number generator.
            * **shuffle** (*bool*) - (True) Whether or not to shuffle the data before splitting.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_results_path, output_coefs_path, output_percent_path, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_coefs_path": output_coefs_path, "output_percent_path": output_percent_path } 
        }

        # Properties specific for BB
        self.independent_vars = properties.get('independent_vars', [])
        self.target = properties.get('target', '')
        self.scale = properties.get('scale', True)
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
        self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", out_log, self.__class__.__name__)
        self.io_dict["out"]["output_results_path"] = check_output_path(self.io_dict["out"]["output_results_path"],"output_results_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_coefs_path"] = check_output_path(self.io_dict["out"]["output_coefs_path"],"output_coefs_path", True, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_percent_path"] = check_output_path(self.io_dict["out"]["output_percent_path"],"output_percent_path", True, out_log, self.__class__.__name__)

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
            output_file_list = [self.io_dict["out"]["output_results_path"],self.io_dict["out"]["output_coefs_path"],self.io_dict["out"]["output_percent_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])

        # variance inflation factor
        # List with all features where we want to check for multicollinearity
        feats = data[self.independent_vars]
        vif_table = pd.DataFrame()
        vif_table["variable"] = feats.columns
        # here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
        vif_table["VIF"] = [variance_inflation_factor(feats.values, i) for i in range(feats.shape[1])]
        fu.log('Calculating variance inflation factor (VIF) for independent variables\n\nVARIANCE INFLATION FACTOR\n\n%s\n' % vif_table, out_log, self.global_log)
        # TODO: SAVE TO TEMPORARY??????

        # declare inputs and targets
        targets = data[self.target]
        # the inputs are everything BUT the target or dependent variable, so we can simply drop it
        inputs = data.drop([self.target],axis=1)

        t_inputs = inputs
        # scale dataset
        if self.scale:
            fu.log('Scaling dataset', out_log, self.global_log)
            scaler = StandardScaler()
            scaler.fit(t_inputs)
            t_inputs = scaler.transform(t_inputs)

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        x_train, x_test, y_train, y_test = train_test_split(t_inputs, targets, test_size=self.test_size, random_state=self.random_state)

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

        # weights table
        weights_table = pd.DataFrame(inputs.columns.values, columns=['variable'])
        weights_table['weight'] = coef
        fu.log('Calculating weights (coefficients) of the regression\n\nWEIGHTS\n\n%s\n' % weights_table, out_log, self.global_log)
        # TODO: SAVE TO TEMPORARY??????

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
        fu.log('Calculating scores and coefficients\n\nR2 AND ADJUSTED R2\n\n%s\n\nP-VALUES\n\n%s\n' % (r2_table, coefs_table), out_log, self.global_log)
        # TODO: SAVE TO TEMPORARY??????
        # coefs_table.to_csv(self.io_dict["out"]["output_coefs_path"], index = False, header=True, float_format='%.3f')

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

        # prediction
        new_data_table = pd.DataFrame(data=get_list_of_predictors(self.predictions),columns=self.independent_vars)
        new_data = new_data_table
        if self.scale:
            new_data = scaler.transform(new_data_table)
        p = reg.predict(new_data)
        p = np.around(p, 2)
        new_data_table[self.target] = p
        fu.log('Saving results to %s\n\nPREDICTION RESULTS\n\n%s\n' % (self.io_dict["out"]["output_results_path"], new_data_table), out_log, self.global_log)
        new_data_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and calculates coefficients and predictions for a linear regression.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--output_coefs_path', required=False, help='Path to the output coefficients file. Accepted formats: csv.')
    parser.add_argument('--output_percent_path', required=False, help='Path to the output percentages table file. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    LinearRegression(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_coefs_path=args.output_coefs_path, 
                   output_percent_path=args.output_percent_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

