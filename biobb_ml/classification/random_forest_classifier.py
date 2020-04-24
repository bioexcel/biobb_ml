#!/usr/bin/env python3

"""Module containing the RandomForestClassifier class and the command line interface."""
import argparse
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, log_loss, accuracy_score
from sklearn import ensemble
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.classification.common import *

class RandomForestClassifier():
    """Trains and tests a given dataset and calculates coefficients and predictions for a random forest classifier.
    Wrapper of the sklearn.ensemble.RandomForestClassifier module
    Visit the 'sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>'_. 

    Args:
        input_dataset_path (str): Path to the input dataset. Accepted formats: csv.
        output_results_path (str): Path to the output results file. Accepted formats: csv.
        output_test_table_path (str) (Optional): Path to the test table file. Accepted formats: csv.
        output_plot_path (str) (Optional): Path to the binary classifier evaluator plot file. Includes confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. Accepted formats: png.
        properties (dic):
            * **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **n_estimators** (*int*) - (100) The number of trees in the forest.
            * **bootstrap** (*bool*) - (True) Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
            * **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
            * **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
            * **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_results_path, output_test_table_path=None, output_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_test_table_path": output_test_table_path, "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.independent_vars = properties.get('independent_vars', [])
        self.target = properties.get('target', '')
        self.n_estimators = properties.get('n_estimators', 100)
        self.bootstrap = properties.get('bootstrap', True)
        self.normalize_cm =  properties.get('normalize_cm', False)
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
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the RandomForestClassifier module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_results_path"],self.io_dict["out"]["output_test_table_path"],self.io_dict["out"]["output_plot_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])

        # declare inputs and targets
        targets = data[self.target]
        # the inputs are all the independent variables
        t_inputs = data.filter(self.independent_vars)

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        x_train, x_test, y_train, y_test = train_test_split(t_inputs, targets, test_size=self.test_size, random_state=4)

        # classification
        fu.log('Training dataset applying decision tree classification', out_log, self.global_log)
        random_forest = ensemble.RandomForestClassifier(n_estimators = self.n_estimators,  bootstrap = self.bootstrap)
        random_forest.fit(x_train,y_train)
        y_hat_train = random_forest.predict(x_train)
        # classification report
        cr_train = classification_report(y_train, y_hat_train)
        # log loss
        yhat_prob_train = random_forest.predict_proba(x_train)
        l_loss_train = log_loss(y_train, yhat_prob_train)
        fu.log('Calculating scores and report for training dataset\n\nCLASSIFICATION REPORT\n\n%s\nLog loss: %.3f\n' % (cr_train, l_loss_train), out_log, self.global_log)

        # compute confusion matrix
        cnf_matrix_train = confusion_matrix(y_train, y_hat_train)
        np.set_printoptions(precision=2)
        if self.normalize_cm:
            cnf_matrix_train = cnf_matrix_train.astype('float') / cnf_matrix_train.sum(axis=1)[:, np.newaxis]
            cm_type = 'NORMALIZED CONFUSION MATRIX'
        else:
            cm_type = 'CONFUSION MATRIX, WITHOUT NORMALIZATION'

        fu.log('Calculating confusion matrix for training dataset\n\n%s\n\n%s\n' % (cm_type, cnf_matrix_train), out_log, self.global_log)

        # testing
        # predict data from x_test
        y_hat_test = random_forest.predict(x_test)
        test_table = pd.DataFrame(y_hat_test, columns=['prediction'])
        # reset y_test (problem with old indexes column)
        y_test = y_test.reset_index(drop=True)
        # add real values to predicted ones in test_table table
        test_table['target'] = y_test
        fu.log('Testing\n\nTEST DATA\n\n%s\n' % test_table, out_log, self.global_log)

        # classification report
        cr = classification_report(y_test, y_hat_test)
        # log loss
        yhat_prob = random_forest.predict_proba(x_test)
        l_loss = log_loss(y_test, yhat_prob)
        fu.log('Calculating scores and report for testing dataset\n\nCLASSIFICATION REPORT\n\n%s\nLog loss: %.3f\n' % (cr, l_loss), out_log, self.global_log)

        # compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_hat_test)
        np.set_printoptions(precision=2)
        if self.normalize_cm:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
            cm_type = 'NORMALIZED CONFUSION MATRIX'
        else:
            cm_type = 'CONFUSION MATRIX, WITHOUT NORMALIZATION'

        fu.log('Calculating confusion matrix for testing dataset\n\n%s\n\n%s\n' % (cm_type, cnf_matrix), out_log, self.global_log)

        if(self.io_dict["out"]["output_test_table_path"]): 
            fu.log('Saving testing data to %s' % self.io_dict["out"]["output_test_table_path"], out_log, self.global_log)
            test_table.to_csv(self.io_dict["out"]["output_test_table_path"], index = False, header=True)

        # plot 
        if self.io_dict["out"]["output_plot_path"]: 
            vs = targets.unique().tolist()
            vs.sort()
            if len(vs) > 2:
                plot = plotMultipleCM(cnf_matrix_train, cnf_matrix, self.normalize_cm, vs)
                fu.log('Saving confusion matrix plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            else:
                plot = plotBinaryClassifier(random_forest, yhat_prob_train, yhat_prob, cnf_matrix_train, cnf_matrix, y_train, y_test, normalize=self.normalize_cm)
                fu.log('Saving binary classifier evaluator plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        # prediction
        new_data_table = pd.DataFrame(data=get_list_of_predictors(self.predictions),columns=self.independent_vars)
        new_data = new_data_table
        p = random_forest.predict(new_data)
        p = np.around(p, 2)

        new_data_table[self.target] = p
        fu.log('Predicting results\n\nPREDICTION RESULTS\n\n%s\n' % new_data_table, out_log, self.global_log)
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        new_data_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and calculates coefficients and predictions for a random forest classifier.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--output_test_table_path', required=False, help='Path to the test table file. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the binary classifier evaluator plot file. Includes confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    RandomForestClassifier(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_test_table_path=args.output_test_table_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

