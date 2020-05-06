#!/usr/bin/env python3

"""Module containing the ClassificationNeuralNetwork class and the command line interface."""
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.neural_networks.common import *
sns.set()

class ClassificationNeuralNetwork():
    """Trains and tests a given dataset and calculates coefficients and predictions for a NN classification.
    Wrapper of the TensorFlow Keras API
    Visit the 'TensorFlow Keras official website <https://www.tensorflow.org/guide/keras>'_. 

    Args:
        input_dataset_path (str): Path to the input dataset. Accepted formats: csv.
        output_results_path (str): Path to the output results file. Accepted formats: csv.
        output_test_table_path (str) (Optional): Path to the test table file. Accepted formats: csv.
        output_plot_path (str) (Optional): Loss, accuracy, MAE and MSE plots. Accepted formats: png.
        properties (dic):
            * **features** (*list*) - (None) Independent variables or columns from your dataset you want to train.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
            * **validation_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the validation split. It should be between 0.0 and 1.0.
            * **test_size** (*float*) - (0.1) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
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
        self.features = properties.get('features', [])
        self.target = properties.get('target', '')
        self.predictions = properties.get('predictions', [])
        self.validation_size = properties.get('validation_size', 0.1)
        self.test_size = properties.get('test_size', 0.1)
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
        """Launches the execution of the ClassificationNeuralNetwork module."""

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

        targets = data[self.target].to_numpy()
        # the inputs are all the independent variables
        inputs = data.filter(self.features)

        fu.log('Scaling dataset', out_log, self.global_log)
        scaled_inputs = scale(inputs)

        shuffled_indices = np.arange(scaled_inputs.shape[0])
        np.random.shuffle(shuffled_indices)

        shuffled_inputs = scaled_inputs[shuffled_indices]
        shuffled_targets = targets[shuffled_indices]

        X_train, X_test, y_train, y_test = train_test_split(shuffled_inputs, shuffled_targets, test_size=self.test_size, random_state=1)
        
        #print(X_train.shape, X_test.shape)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(self.validation_size/(1-self.test_size)), random_state=1)

        # Set the input and output sizes
        output_size = np.unique(y_train).size
        # Use same hidden layer size for both hidden layers. Not a necessity.
        hidden_layer_size = 50

        # TODO
        # LOGS
        # custom hidden_layer_size & activation (S45_L312/Course-Notes-Section-6.pdf) for n layers (dictionary)
        # create n layers in for loop: http://localhost:8888/notebooks/S50_L351/TensorFlow_MNIST_All_Exercises.ipynb
        # activation for output layer
        # COMPILE: optimizer & loss
        # SUMMARY
        # TEST TABLE (test values + predicted values in (0,1) probability column)
        # SCORES TABLE (accuracies, losses, etc)
        # RESULTS TABLE (results in yml file)
        # PLOTS
            
        # define how the model will look like
        model = tf.keras.Sequential([
            # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
            # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
            tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(X_train.shape[1],)), # 1st hidden layer
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
            # the final layer is no different, we just make sure to activate it with softmax
            tf.keras.layers.Dense(output_size, activation='softmax') # output layer
        ])

        ### Choose the optimizer and the loss function

        # we define the optimizer we'd like to use, 
        # the loss function, 
        # and the metrics we are interested in obtaining at each iteration
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy','mae', 'mse'])

        #print(model.summary())

        ### Training
        # That's where we train the model we have built.

        # set the batch size
        batch_size = 100

        # set a maximum number of training epochs
        max_epochs = 100

        # set an early stopping mechanism
        # let's set patience=2, to be a bit tolerant against random validation loss increases
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

        # fit the model
        # note that this time the train, validation and test data are not iterable
        mf = model.fit(X_train, # train inputs
          y_train, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping], # early stopping
          validation_data=(X_val, y_val), # validation data
          verbose = 1 # making sure we get enough information about the training process
          )

        print(model.summary())

        print(mf.history)

        exit()















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
        x_train, x_test, y_train, y_test = train_test_split(t_inputs, targets, test_size=self.test_size, random_state=5)

        # regression
        fu.log('Training dataset applying linear regression', out_log, self.global_log)
        reg = linear_model.ClassificationNeuralNetwork()
        reg.fit(x_train, y_train)

        # scores and coefficients train
        y_hat_train = reg.predict(x_train)
        rmse = (np.sqrt(mean_squared_error(y_train, y_hat_train)))
        rss = np.mean((y_hat_train - y_train) ** 2)
        score = reg.score(x_train, y_train)
        bias = reg.intercept_
        coef = reg.coef_
        coef = [ '%.3f' % item for item in coef ]
        adj_r2 = adjusted_r2(x_train, y_train, score)
        p_values = f_regression(x_train, y_train)[1]
        p_values = [ '%.3f' % item for item in p_values ]

        # r-squared
        r2_table = pd.DataFrame()
        r2_table["feature"] = ['R2','Adj. R2', 'RMSE', 'RSS']
        r2_table['coefficient'] = [score, adj_r2, rmse, rss]

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
        fu.log('Calculating scores and coefficients for training dataset\n\nR2, ADJUSTED R2 & RMSE\n\n%s\n\nCOEFFS & P-VALUES\n\n%s\n' % (r2_table, coefs_table), out_log, self.global_log)

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
        
        # scores and coefficients test
        r2_test = reg.score(x_test, y_test)
        adj_r2_test = adjusted_r2(x_test, y_test, r2_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_hat_test))
        rss_test = np.mean((y_hat_test - y_test) ** 2)

        # r-squared
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        r2_table_test = pd.DataFrame()
        r2_table_test["feature"] = ['R2','Adj. R2', 'RMSE', 'RSS']
        r2_table_test['coefficient'] = [r2_test, adj_r2_test, rmse_test, rss_test]

        fu.log('Calculating scores and coefficients for testing dataset\n\nR2, ADJUSTED R2 & RMSE\n\n%s\n' % r2_table_test, out_log, self.global_log)

        if(self.io_dict["out"]["output_test_table_path"]): 
            fu.log('Saving testing data to %s' % self.io_dict["out"]["output_test_table_path"], out_log, self.global_log)
            test_table.to_csv(self.io_dict["out"]["output_test_table_path"], index = False, header=True)

        # create test plot
        if(self.io_dict["out"]["output_plot_path"]): 
            fu.log('Saving residual plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot = plotResults(y_train, y_hat_train, y_test, y_hat_test)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)


        # prediction
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
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
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and calculates coefficients and predictions for a NN classification.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--output_test_table_path', required=False, help='Path to the test table file. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Loss, accuracy, MAE and MSE plots. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    ClassificationNeuralNetwork(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_test_table_path=args.output_test_table_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

