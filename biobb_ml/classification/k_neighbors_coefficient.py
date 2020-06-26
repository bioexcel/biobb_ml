#!/usr/bin/env python3

"""Module containing the KNeighborsCoefficient class and the command line interface."""
import argparse
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss
from sklearn.neighbors import KNeighborsClassifier
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.classification.common import *


class KNeighborsCoefficient():
    """Trains and tests a given dataset and calculates best K coefficient for a k-nearest neighbors classification.
    Wrapper of the sklearn.neighbors.KNeighborsClassifier module
    Visit the `sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_k_neighbors_coefficient.csv>`_. Accepted formats: csv.
        output_results_path (str): Path to the accuracy values list. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_k_neighbors_coefficient.csv>`_. Accepted formats: csv.
        output_plot_path (str) (Optional): Path to the accuracy plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_k_neighbors_coefficient.png>`_. Accepted formats: png.
        properties (dic):
            * **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **metric** (*string*) - ("minkowski") The distance metric to use for the tree. Values: euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobi.
            * **max_neighbors** (*int*) - (6) Maximum number of neighbors to use by default for kneighbors queries.
            * **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
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
        self.independent_vars = properties.get('independent_vars', [])
        self.target = properties.get('target', '')
        self.metric = properties.get('metric', 'minkowski')
        self.max_neighbors = properties.get('max_neighbors', 6)
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
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the KNeighborsCoefficient module."""

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

        # declare inputs and targets
        targets = data[self.target]
        # the inputs are all the independent variables
        inputs = data.filter(self.independent_vars)

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=self.test_size, random_state=4)

        # scale dataset
        fu.log('Scaling dataset', out_log, self.global_log)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(x_train)

        # training and getting accuracy for each K
        fu.log('Training dataset applying k neighbors classification from 1 to %d n_neighbors' % self.max_neighbors, out_log, self.global_log)
        neighbors = np.arange(1,self.max_neighbors + 1)
        train_accuracy =np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))
        std_acc = np.zeros((self.max_neighbors))
        X_test = scaler.transform(x_test)
        for i,k in enumerate(neighbors):
            #Setup a knn classifier with k neighbors
            model = KNeighborsClassifier(n_neighbors = k)
            #Fit the model
            model.fit(X_train, y_train)
            #Compute accuracy on the training set
            train_accuracy[i] = model.score(X_train, y_train)
            #Compute accuracy on the test set
            test_accuracy[i] = model.score(X_test, y_test)
            # deviation
            yhat_test = model.predict(X_test)
            std_acc[i - 1] = np.std(yhat_test == y_test) / np.sqrt(yhat_test.shape[0])

        # best K / best accuracy
        best_k = test_accuracy.argmax() + 1
        best_accuracy = test_accuracy.max()

        # accuracy table
        test_table_accuracy = pd.DataFrame(data={'K': np.arange(1, self.max_neighbors + 1), 'accuracy': test_accuracy})
        fu.log('Calculating accuracy for each K\n\nACCURACY\n\n%s\n' % test_table_accuracy.to_string(index=False), out_log, self.global_log)

        # classification report
        cr_test = classification_report(y_test, model.predict(X_test))
        # log loss
        yhat_prob = model.predict_proba(X_test)
        l_loss = log_loss(y_test, yhat_prob)
        fu.log('Calculating report for testing dataset and best K = %d | accuracy = %.3f\n\nCLASSIFICATION REPORT\n\n%s\nLog loss: %.3f\n' % (best_k, best_accuracy, cr_test, l_loss), out_log, self.global_log)

        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        test_table_accuracy.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        # accuracy plot
        if self.io_dict["out"]["output_plot_path"]: 
            fu.log('Saving accuracy plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plt.title('k-NN Varying number of neighbors')
            plt.fill_between(range(1, self.max_neighbors + 1), test_accuracy - std_acc, test_accuracy + std_acc, alpha = 0.10)
            plt.plot(neighbors, train_accuracy)
            plt.plot(neighbors, test_accuracy)
            plt.axvline(x=best_k, c='red')
            plt.legend(('Training Accuracy', 'Testing accuracy', 'Best K', '+/- 3xstd'))
            plt.xlabel('Number of neighbors')
            plt.ylabel('Accuracy')
            plt.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)
            plt.tight_layout()

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and calculates best K coefficient for a k-nearest neighbors classification.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the accuracy values list. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the accuracy plot. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    KNeighborsCoefficient(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

