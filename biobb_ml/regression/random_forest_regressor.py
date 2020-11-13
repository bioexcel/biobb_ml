#!/usr/bin/env python3

"""Module containing the RandomForestRegressor class and the command line interface."""
import argparse
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.regression.common import *
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
sns.set()


class RandomForestRegressor():
    """Trains and tests a given dataset and saves the model and scaler for a random forest regressor.
    Wrapper of the sklearn.ensemble.RandomForestRegressor module
    Visit the `sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/dataset_random_forest_regressor.csv>`_. Accepted formats: csv.
        output_model_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_model_random_forest_regressor.pkl>`_. Accepted formats: pkl.
        output_test_table_path (str) (Optional): Path to the test table file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_test_random_forest_regressor.csv>`_. Accepted formats: csv.
        output_plot_path (str) (Optional): Residual plot checks the error between actual values and predicted values. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_random_forest_regressor.png>`_. Accepted formats: png.
        properties (dic):
            * **independent_vars** (*dict*) - (None) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **target** (*dict*) - (None) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **weight** (*dict*) - (None) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **scale** (*bool*) - (False) Whether or not to scale the input dataset.
            * **n_estimators** (*int*) - (10) The number of trees in the forest.
            * **max_depth** (*int*) - (None) The maximum depth of the tree.
            * **random_state** (*int*) - (5) Controls the shuffling applied to the data before applying the split. .
            * **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_model_path, output_test_table_path=None, output_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_model_path": output_model_path, "output_test_table_path": output_test_table_path, "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.independent_vars = properties.get('independent_vars', None)
        self.target = properties.get('target', None)
        self.weight = properties.get('weight', None)
        self.n_estimators = properties.get('n_estimators', 10)
        self.max_depth = properties.get('max_depth', None)
        self.random_state = properties.get('random_state', 5)
        self.test_size = properties.get('test_size', 0.2)
        self.scale = properties.get('scale', False)
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
        self.io_dict["out"]["output_model_path"] = check_output_path(self.io_dict["out"]["output_model_path"],"output_model_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_test_table_path"] = check_output_path(self.io_dict["out"]["output_test_table_path"],"output_test_table_path", True, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the RandomForestRegressor module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_model_path"],self.io_dict["out"]["output_test_table_path"],self.io_dict["out"]["output_plot_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        if 'columns' in self.independent_vars:
            header = 0
        else:
            header = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = header, sep="\s+|;|:|,|\t", engine="python")
        #pd.set_option('display.float_format', lambda x: '%.6f' % x)

        # declare inputs, targets and weights
        # the inputs are all the independent variables
        X = getIndependentVars(self.independent_vars, data, out_log, self.__class__.__name__)
        # target
        y = getTarget(self.target, data, out_log, self.__class__.__name__)
        # weights
        if self.weight:
            w = getWeight(self.weight, data, out_log, self.__class__.__name__)

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        arrays_sets = (X, y)
        # if user provide weights
        if self.weight:
            arrays_sets = arrays_sets + (w,)
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(*arrays_sets, test_size=self.test_size, random_state = self.random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(*arrays_sets, test_size=self.test_size, random_state = self.random_state)


        # scale dataset
        if self.scale: 
            fu.log('Scaling dataset', out_log, self.global_log)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            ## PUT HERE SCALER FOR TARGET????

        # regression
        fu.log('Training dataset applying random forest regressor', out_log, self.global_log)
        #model = ensemble.RandomForestRegressor(max_depth = self.max_depth, n_estimators = self.n_estimators)
        model = ensemble.RandomForestRegressor(max_depth = self.max_depth, n_estimators = self.n_estimators, random_state = self.random_state)
        arrays_fit = (X_train, y_train)
        # if user provide weights
        if self.weight:
            arrays_fit = arrays_fit + (w_train,)

        model.fit(*arrays_fit)

        # scores and coefficients train
        y_hat_train = model.predict(X_train)
        rmse = (np.sqrt(mean_squared_error(y_train, y_hat_train)))
        rss = ((y_train - y_hat_train) ** 2).sum()
        score_train_inputs = (y_train, y_hat_train)
        if self.weight:
            score_train_inputs = score_train_inputs + (w_train,)
        score = r2_score(*score_train_inputs)
        #adj_r2 = adjusted_r2(X_train, y_train, score)

        # r-squared
        r2_table = pd.DataFrame()
        #r2_table["feature"] = ['R2','Adj. R2', 'RMSE', 'RSS']
        #r2_table['coefficient'] = [score, adj_r2, rmse, rss]
        r2_table["feature"] = ['R2','RMSE', 'RSS']
        r2_table['coefficient'] = [score, rmse, rss]

        fu.log('Calculating scores and coefficients for TRAINING dataset\n\nSCORES\n\n%s\n' % r2_table, out_log, self.global_log)


        # testing
        # predict data from x_test
        if self.scale:
            X_test = scaler.transform(X_test)
        y_hat_test = model.predict(X_test)
        test_table = pd.DataFrame(y_hat_test, columns=['prediction'])
        # reset y_test (problem with old indexes column)
        y_test = y_test.reset_index(drop=True)
        # add real values to predicted ones in test_table table
        test_table['target'] = y_test
        # calculate difference between target and prediction (absolute and %)
        test_table['residual'] = test_table['target'] - test_table['prediction']
        test_table['difference %'] = np.absolute(test_table['residual']/test_table['target']*100)
        #pd.set_option('display.float_format', lambda x: '%.2f' % x)
        # sort by difference in %
        test_table = test_table.sort_values(by=['difference %'])
        test_table = test_table.reset_index(drop=True)
        fu.log('Testing\n\nTEST DATA\n\n%s\n' % test_table, out_log, self.global_log)

        # scores and coefficients test
        score_test_inputs = (y_test, y_hat_test)
        if self.weight:
            score_test_inputs = score_test_inputs + (w_test,)
        r2_test = r2_score(*score_test_inputs)
        #adj_r2_test = adjusted_r2(X_test, y_test, r2_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_hat_test))
        rss_test = ((y_test - y_hat_test) ** 2).sum()

        # r-squared
        #pd.set_option('display.float_format', lambda x: '%.6f' % x)
        r2_table_test = pd.DataFrame()
        #r2_table_test["feature"] = ['R2','Adj. R2', 'RMSE', 'RSS']
        #r2_table_test['coefficient'] = [r2_test, adj_r2_test, rmse_test, rss_test]
        r2_table_test["feature"] = ['R2','RMSE', 'RSS']
        r2_table_test['coefficient'] = [r2_test, rmse_test, rss_test]

        fu.log('Calculating scores and coefficients for TESTING dataset\n\nSCORES\n\n%s\n' % r2_table_test, out_log, self.global_log)

        if(self.io_dict["out"]["output_test_table_path"]): 
            fu.log('Saving testing data to %s' % self.io_dict["out"]["output_test_table_path"], out_log, self.global_log)
            test_table.to_csv(self.io_dict["out"]["output_test_table_path"], index = False, header=True)

        # create test plot
        if(self.io_dict["out"]["output_plot_path"]): 
            fu.log('Saving residual plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            y_hat_test = y_hat_test.flatten()
            y_hat_train = y_hat_train.flatten()
            plot = plotResults(y_train, y_hat_train, y_test, y_hat_test)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        # save model, scaler and parameters
        # MODIFY
        variables = {
            'target': self.target,
            'independent_vars': self.independent_vars
        }
        fu.log('Saving model to %s' % self.io_dict["out"]["output_model_path"], out_log, self.global_log)
        with open(self.io_dict["out"]["output_model_path"], "wb") as f:
            joblib.dump(model, f)
            if self.scale: joblib.dump(scaler, f)
            joblib.dump(variables, f)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and saves the model and scaler for a random forest regressor.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_model_path', required=True, help='Path to the output model file. Accepted formats: pkl.')
    parser.add_argument('--output_test_table_path', required=False, help='Path to the test table file. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Residual plot checks the error between actual values and predicted values. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    RandomForestRegressor(input_dataset_path=args.input_dataset_path,
                   output_model_path=args.output_model_path, 
                   output_test_table_path=args.output_test_table_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

