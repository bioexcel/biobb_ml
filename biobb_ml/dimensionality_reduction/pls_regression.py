#!/usr/bin/env python3

"""Module containing the PLS_Regression class and the command line interface."""
import argparse
import io
import warnings
from sys import stdout
#from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.dimensionality_reduction.common import *


class PLS_Regression():
    """Gives results for a Partial Least Square (PLS) Regression.
    Wrapper of the sklearn.cross_decomposition.PLSRegression module
    Visit the `sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html>`_. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/dimensionality_reduction/dataset_pls_regression.csv>`_. Accepted formats: csv.
        output_results_path (str): Table with R2 and MSE for calibration and cross-validation data. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_results_pls_regression.csv>`_. Accepted formats: csv.
        output_plot_path (str) (Optional): Path to the R2 cross-validation plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_plot_pls_regression.png>`_. Accepted formats: png.
        properties (dic):
            * **features** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **n_components** (*int*) - (5) Maximum number of components to use by default for PLS queries.
            * **cv** (*int*) - (10) Specify the number of folds in the cross-validation splitting strategy. Value must be betwwen 2 and number of samples in the dataset.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_results_path, output_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.features = properties.get('features', [])
        self.target = properties.get('target', '')
        self.n_components = properties.get('n_components', 5)
        self.cv = properties.get('cv', 10)
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
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    def warn(*args, **kwargs):
        pass

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the PLS_Regression module."""

        # trick for disable warnings in interations
        warnings.warn = self.warn

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_results_path"],self.io_dict["out"]["output_plot_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])

        # get targets
        y = data[self.target]
        # get features
        features = data.filter(self.features)

        # get rid of baseline and linear variations calculating second derivative
        # fu.log('Performing second derivative on the data', out_log, self.global_log)
        # self.window_length = getWindowLength(17, features.shape[1])
        # X = savgol_filter(features, window_length = self.window_length, polyorder = 2, deriv = 2)
        X = features

        # define PLS object with optimal number of components
        model = PLSRegression(n_components = self.n_components)
        # fit to the entire dataset
        model.fit(X, y)
        y_c = model.predict(X)
        # cross-validation
        y_cv = cross_val_predict(model, X, y, cv = self.cv)
        # calculate scores for calibration and cross-validation
        score_c = r2_score(y, y_c)
        score_cv = r2_score(y, y_cv)
        # calculate mean squared error for calibration and cross validation
        mse_c = mean_squared_error(y, y_c)
        mse_cv = mean_squared_error(y, y_cv)
        # create scores table
        r2_table = pd.DataFrame()
        r2_table["feature"] = ['R2 calib','R2 CV', 'MSE calib', 'MSE CV']
        r2_table['coefficient'] = [score_c, score_cv, mse_c, mse_cv]

        fu.log('Generating scores table\n\nR2 & MSE TABLE\n\n%s\n' % r2_table, out_log, self.global_log)

        # save results table
        fu.log('Saving R2 & MSE table to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        r2_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        # mse plot
        if self.io_dict["out"]["output_plot_path"]: 
            fu.log('Saving MSE plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot = PLSRegPlot(y, y_c, y_cv)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Gives results for a Partial Least Square (PLS) Regression.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Table with R2 and MSE for calibration and cross-validation data. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the R2 cross-validation plot. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    PLS_Regression(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()
