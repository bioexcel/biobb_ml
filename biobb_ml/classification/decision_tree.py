#!/usr/bin/env python3

"""Module containing the DecisionTree class and the command line interface."""
import argparse
import io
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.classification.common import *

class DecisionTree():
    """
    | biobb_ml DecisionTree
    | Wrapper of the scikit-learn DecisionTreeClassifier method. 
    | Trains and tests a given dataset and saves the model and scaler. Visit the `DecisionTreeClassifier documentation page <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_ in the sklearn official website for further information. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_decision_tree.csv>`_. Accepted formats: csv (edam:format_3752).
        output_model_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_decision_tree.pkl>`_. Accepted formats: pkl (edam:format_3653).
        output_test_table_path (str) (Optional): Path to the test table file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_decision_tree.csv>`_. Accepted formats: csv (edam:format_3752).
        output_plot_path (str) (Optional): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_decision_tree.png>`_. Accepted formats: png (edam:format_3603).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **independent_vars** (*dict*) - ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **target** (*dict*) - ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **weight** (*dict*) - ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **criterion** (*string*) - ("gini") The function to measure the quality of a split. Values: gini (for the Gini impurity), entropy (for the information gain).
            * **max_depth** (*int*) - (4) [1~100|1] The maximum depth of the model. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            * **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
            * **random_state_method** (*int*) - (5) [1~1000|1] Controls the randomness of the estimator.
            * **random_state_train_test** (*int*) - (5) [1~1000|1] Controls the shuffling applied to the data before applying the split.
            * **test_size** (*float*) - (0.2) [0~1|0.05] Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
            * **scale** (*bool*) - (False) Whether or not to scale the input dataset.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.classification.decision_tree import decision_tree
            prop = { 
                'independent_vars': { 
                    'columns': [ 'column1', 'column2', 'column3' ] 
                }, 
                'target': { 
                    'column': 'target' 
                }, 
                'criterion': 'entropy', 
                'test_size': 0.2 
            }
            decision_tree(input_dataset_path='/path/to/myDataset.csv', 
                            output_model_path='/path/to/newModel.pkl', 
                            output_test_table_path='/path/to/newTable.csv', 
                            output_plot_path='/path/to/newPlot.png', 
                            properties=prop)


    Info:
        * wrapped_software:
            * name: scikit-learn DecisionTreeClassifier
            * version: >=0.23.1
            * license: BSD 3-Clause
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_dataset_path, output_model_path, 
                output_test_table_path=None, output_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_model_path": output_model_path, "output_test_table_path": output_test_table_path, "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.independent_vars = properties.get('independent_vars', {})
        self.target = properties.get('target', {})
        self.weight = properties.get('weight', {})
        self.criterion = properties.get('criterion', 'gini')
        self.max_depth = properties.get('max_depth', 4)
        self.normalize_cm =  properties.get('normalize_cm', False)
        self.random_state_method = properties.get('random_state_method', 5)
        self.random_state_train_test = properties.get('random_state_train_test', 5)
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
        if self.io_dict["out"]["output_test_table_path"]:
            self.io_dict["out"]["output_test_table_path"] = check_output_path(self.io_dict["out"]["output_test_table_path"],"output_test_table_path", True, out_log, self.__class__.__name__)
        if self.io_dict["out"]["output_plot_path"]:
            self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`DecisionTree <classification.decision_tree.DecisionTree>` classification.decision_tree.DecisionTree object."""

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
            labels = getHeader(self.io_dict["in"]["input_dataset_path"])
            skiprows = 1
        else:
            labels = None
            skiprows = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = None, sep="\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)

        # declare inputs, targets and weights
        # the inputs are all the independent variables
        X = getIndependentVars(self.independent_vars, data, out_log, self.__class__.__name__)
        fu.log('Independent variables: [%s]' % (getIndependentVarsList(self.independent_vars)), out_log, self.global_log)
        # target
        y = getTarget(self.target, data, out_log, self.__class__.__name__)
        fu.log('Target: %s' % (getTargetValue(self.target)), out_log, self.global_log)
        # weights
        if self.weight:
            w = getWeight(self.weight, data, out_log, self.__class__.__name__)
            fu.log('Weight column provided', out_log, self.global_log)

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        arrays_sets = (X, y)
        # if user provide weights
        if self.weight:
            arrays_sets = arrays_sets + (w,)
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(*arrays_sets, test_size=self.test_size, random_state = self.random_state_train_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(*arrays_sets, test_size=self.test_size, random_state = self.random_state_train_test)

        # scale dataset
        if self.scale: 
            fu.log('Scaling dataset', out_log, self.global_log)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        # classification
        fu.log('Training dataset applying decision tree classification', out_log, self.global_log)
        model = DecisionTreeClassifier(criterion = self.criterion, max_depth = self.max_depth, random_state = self.random_state_method)
        arrays_fit = (X_train, y_train)
        # if user provide weights
        if self.weight:
            arrays_fit = arrays_fit + (w_train,)

        model.fit(*arrays_fit)

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
        if self.scale:
            X_test = scaler.transform(X_test)
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
            vs = y.unique().tolist()
            vs.sort()
            if len(vs) > 2:
                plot = plotMultipleCM(cnf_matrix_train, cnf_matrix, self.normalize_cm, vs)
                fu.log('Saving confusion matrix plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            else:
                plot = plotBinaryClassifier(model, yhat_prob_train, yhat_prob, cnf_matrix_train, cnf_matrix, y_train, y_test, normalize=self.normalize_cm)
                fu.log('Saving binary classifier evaluator plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        # save model, scaler and parameters
        tv = y.unique().tolist()
        tv.sort()
        variables = {
            'target': self.target,
            'independent_vars': self.independent_vars,
            'scale': self.scale,
            'target_values': tv
        }
        fu.log('Saving model to %s' % self.io_dict["out"]["output_model_path"], out_log, self.global_log)
        with open(self.io_dict["out"]["output_model_path"], "wb") as f:
            joblib.dump(model, f)
            if self.scale: joblib.dump(scaler, f)
            joblib.dump(variables, f)

        return 0

def decision_tree(input_dataset_path: str, output_model_path: str, output_test_table_path: str = None, output_plot_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`DecisionTree <classification.decision_tree.DecisionTree>` class and
    execute the :meth:`launch() <classification.decision_tree.DecisionTree.launch>` method."""

    return DecisionTree(input_dataset_path=input_dataset_path, 
                   output_model_path=output_model_path, 
                   output_test_table_path=output_test_table_path, 
                   output_plot_path=output_plot_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of the scikit-learn DecisionTreeClassifier method. ", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
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
    decision_tree(input_dataset_path=args.input_dataset_path,
                   output_model_path=args.output_model_path, 
                   output_test_table_path=args.output_test_table_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties)

if __name__ == '__main__':
    main()

