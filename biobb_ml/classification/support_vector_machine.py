#!/usr/bin/env python3

"""Module containing the SupportVectorMachine class and the command line interface."""
import argparse
import io
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn import svm
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.classification.common import *

class SupportVectorMachine():
    """Trains and tests a given dataset and saves the model and scaler for a support vector machine.
    Wrapper of the sklearn.svm.SVC module
    Visit the `sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_support_vector_machine.csv>`_. Accepted formats: csv.
        output_model_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_support_vector_machine.pkl>`_. Accepted formats: pkl.
        output_test_table_path (str) (Optional): Path to the test table file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_support_vector_machine.csv>`_. Accepted formats: csv.
        output_plot_path (str) (Optional): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_support_vector_machine.png>`_. Accepted formats: png.
        properties (dic):
            * **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **kernel** (*string*) - ("rbf") Specifies the kernel type to be used in the algorithm. Values: linear, poly, rbf, sigmoid, precomputed.
            * **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
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
        self.independent_vars = properties.get('independent_vars', [])
        self.target = properties.get('target', '')
        self.kernel = properties.get('kernel', 'rbf')
        self.normalize_cm =  properties.get('normalize_cm', False)
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
        self.io_dict["out"]["output_model_path"] = check_output_path(self.io_dict["out"]["output_model_path"],"output_model_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_test_table_path"] = check_output_path(self.io_dict["out"]["output_test_table_path"],"output_test_table_path", True, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the SupportVectorMachine module."""

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
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])

        # declare inputs and targets
        targets = data[self.target]
        # the inputs are all the independent variables
        t_inputs = data.filter(self.independent_vars)

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        x_train, x_test, y_train, y_test = train_test_split(t_inputs, targets, test_size=self.test_size, random_state=4)

        # scale dataset
        fu.log('Scaling dataset', out_log, self.global_log)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(x_train)

        # classification
        fu.log('Training dataset applying support vector machine', out_log, self.global_log)
        model = svm.SVC(kernel = self.kernel, probability = True)
        model.fit(X_train,y_train)
        y_hat_train = model.predict(X_train)
        # classification report
        cr_train = classification_report(y_train, y_hat_train)
        # log loss
        yhat_prob_train = model.predict_proba(X_train)
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
        X_test = scaler.transform(x_test)
        y_hat_test = model.predict(X_test)
        test_table = pd.DataFrame()
        y_hat_prob = model.predict_proba(X_test)
        y_hat_prob = np.around(y_hat_prob, decimals=2)
        y_hat_prob = tuple(map(tuple, y_hat_prob))
        test_table['P' + np.array2string(np.unique(y_test))] = y_hat_prob
        y_test = y_test.reset_index(drop=True)
        test_table['target'] = y_test
        fu.log('Testing\n\nTEST DATA\n\n%s\n' % test_table, out_log, self.global_log)

        # classification report
        cr = classification_report(y_test, y_hat_test)
        # log loss
        yhat_prob = model.predict_proba(X_test)
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
                plot = plotBinaryClassifier(model, yhat_prob_train, yhat_prob, cnf_matrix_train, cnf_matrix, y_train, y_test, normalize=self.normalize_cm)
                fu.log('Saving binary classifier evaluator plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        # save model, scaler and parameters
        tv = targets.unique().tolist()
        tv.sort()
        variables = {
            'target': self.target,
            'independent_vars': self.independent_vars,
            'target_values': tv
        }
        fu.log('Saving model to %s' % self.io_dict["out"]["output_model_path"], out_log, self.global_log)
        with open(self.io_dict["out"]["output_model_path"], "wb") as f:
            joblib.dump(model, f)
            joblib.dump(scaler, f)
            joblib.dump(variables, f)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and saves the model and scaler for a support vector machine.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_model_path', required=True, help='Path to the output model file. Accepted formats: pkl.')
    parser.add_argument('--output_test_table_path', required=False, help='Path to the test table file. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    SupportVectorMachine(input_dataset_path=args.input_dataset_path,
                   output_model_path=args.output_model_path, 
                   output_test_table_path=args.output_test_table_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

