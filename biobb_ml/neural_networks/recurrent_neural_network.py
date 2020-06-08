#!/usr/bin/env python3

"""Module containing the RecurrentNeuralNetwork class and the command line interface."""
import argparse
import h5py
import json
from tensorflow.python.keras.saving import hdf5_format
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# from tensorflow.keras.callbacks import EarlyStopping
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.neural_networks.common import *


class RecurrentNeuralNetwork():
    """Trains and tests a given dataset and save the complete model for a Recurrent Neural Network.
    Wrapper of the TensorFlow Keras Long Short-Term Memory layer
    Visit the 'TensorFlow official website <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>'_. 

    Args:
        input_dataset_path (str): Path to the input dataset. Accepted formats: csv.
        output_model_path (str): Path to the output model file. Accepted formats: h5.
        output_test_table_path (str) (Optional): Path to the test table file. Accepted formats: csv.
        output_plot_path (str) (Optional): Loss, accuracy and MSE plots. Accepted formats: png.
        properties (dic):
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **validation_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the validation split. It should be between 0.0 and 1.0.
            * **window_size** (*int*) - (5) Number of steps for each window of training model.
            * **test_size** (*int*) - (0.1) Represents the number of samples of the dataset to include in the test split.
            * **hidden_layers** (*list*) - (None)  List of dictionaries with hidden layers values. Format: [ { 'size': 50, 'activation': 'relu' } ].
            * **optimizer** (*string*) - ("Adam") Name of optimizer instance. Values: Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD.
            * **learning_rate** (*float*) - (0.02) Determines the step size at each iteration while moving toward a minimum of a loss function
            * **batch_size** (*int*) - (100) Number of samples per gradient update.
            * **max_epochs** (*int*) - (100) Number of epochs to train the model. As the early stopping is enabled, this is a maximum.
            * **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
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
        self.target = properties.get('target', '')
        self.validation_size = properties.get('validation_size', 0.1)
        self.window_size = properties.get('window_size', 5)
        self.test_size = properties.get('test_size', 5)
        self.hidden_layers = properties.get('hidden_layers', [])
        self.optimizer = properties.get('optimizer', 'Adam')
        self.learning_rate = properties.get('learning_rate', 0.02)
        self.batch_size = properties.get('batch_size', 100)
        self.max_epochs = properties.get('max_epochs', 100)
        self.normalize_cm =  properties.get('normalize_cm', False)
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
        self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_model_path"] = check_output_path(self.io_dict["out"]["output_model_path"],"output_model_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_test_table_path"] = check_output_path(self.io_dict["out"]["output_test_table_path"],"output_test_table_path", True, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    def build_model(self, input_shape):
        # create model
        model = Sequential([])

        # if no hidden_layers provided, create manually a hidden layer with default values
        if not self.hidden_layers:
            self.hidden_layers = [ { 'size': 50, 'activation': 'relu' } ]

        # generate hidden_layers
        for i,layer in enumerate(self.hidden_layers):
            if i == 0:
                model.add(LSTM(layer['size'], activation=layer['activation'], kernel_initializer='he_normal', input_shape=input_shape)) # 1st hidden layer
            else:
                model.add(Dense(layer['size'], activation = layer['activation'], kernel_initializer='he_normal'))

        model.add(Dense(1)) # output layer

        return model

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the RecurrentNeuralNetwork module."""

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

        # get target column
        target = data[self.target].to_numpy()

        # split into samples
        X, y = split_sequence(target, self.window_size)
        # reshape into [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        X_train, X_test, y_train, y_test = X[:-self.test_size], X[-self.test_size:], y[:-self.test_size], y[-self.test_size:]

        # build model
        fu.log('Building model', out_log, self.global_log)
        model = self.build_model((X_train.shape[1],1))

        # model summary
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        fu.log('Model summary:\n\n%s\n' % model_summary, out_log, self.global_log)

        # get optimizer
        mod = __import__('tensorflow.keras.optimizers', fromlist = [self.optimizer])
        opt_class = getattr(mod, self.optimizer)
        opt = opt_class(lr = self.learning_rate)
        # compile model
        model.compile(optimizer = opt, loss = 'mse', metrics = ['mse', 'mae'])

        # fitting
        fu.log('Training model', out_log, self.global_log)
        # set an early stopping mechanism
        # set patience=2, to be a bit tolerant against random validation loss increases
        #early_stopping = EarlyStopping(patience=2)
        # fit the model
        mf = model.fit(X_train, 
                       y_train, 
                       batch_size=self.batch_size, 
                       epochs=self.max_epochs, 
                       #callbacks=[early_stopping],
                       validation_split=self.validation_size,
                       verbose = 1)

        train_metrics = pd.DataFrame()
        train_metrics['metric'] = ['Train loss',' Train MAE', 'Train MSE', 'Validation loss', 'Validation MAE', 'Validation MSE']
        train_metrics['coefficient'] = [mf.history['loss'][-1], mf.history['mae'][-1], mf.history['mse'][-1], mf.history['val_loss'][-1], mf.history['val_mae'][-1], mf.history['val_mse'][-1]]

        fu.log('Training metrics\n\nTRAINING METRICS TABLE\n\n%s\n' % train_metrics, out_log, self.global_log)

        # testing
        fu.log('Testing model', out_log, self.global_log)
        test_loss, test_mse, test_mae = model.evaluate(X_test, y_test)

        # predict data from X_test
        test_predictions = model.predict(X_test)
        test_predictions = np.around(test_predictions, decimals=2)        
        tpr = np.squeeze(np.asarray(test_predictions))

        test_metrics = pd.DataFrame()
        test_metrics['metric'] = ['Test loss', 'Test MAE', 'Test MSE']
        test_metrics['coefficient'] = [test_loss, test_mae, test_mse]

        fu.log('Testing metrics\n\nTESTING METRICS TABLE\n\n%s\n' % test_metrics, out_log, self.global_log)

        test_table = pd.DataFrame()
        test_table['prediction'] = tpr
        test_table['target'] = y_test
        test_table['residual'] = test_table['target'] - test_table['prediction']
        test_table['difference %'] = np.absolute(test_table['residual']/test_table['target']*100)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        # sort by difference in %
        test_table = test_table.sort_values(by=['difference %'])
        test_table = test_table.reset_index(drop=True)
        fu.log('TEST DATA\n\n%s\n' % test_table, out_log, self.global_log)

        # save test data
        if(self.io_dict["out"]["output_test_table_path"]): 
            fu.log('Saving testing data to %s' % self.io_dict["out"]["output_test_table_path"], out_log, self.global_log)
            test_table.to_csv(self.io_dict["out"]["output_test_table_path"], index = False, header=True)

        # create test plot
        if(self.io_dict["out"]["output_plot_path"]): 
            fu.log('Saving plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            test_predictions = test_predictions.flatten()
            train_predictions = model.predict(X_train).flatten()
            plot = plotResultsReg(mf.history, y_test, test_predictions, y_train, train_predictions)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        # save model and parameters
        vars_obj = {
            'target': self.target,
            'window_size': self.window_size,
            'type': 'recurrent'
        }
        variables = json.dumps(vars_obj)
        fu.log('Saving model to %s' % self.io_dict["out"]["output_model_path"], out_log, self.global_log)
        with h5py.File(self.io_dict["out"]["output_model_path"], mode='w') as f:
            hdf5_format.save_model_to_hdf5(model, f)
            f.attrs['variables'] = variables

        return 0

def main():
    parser = argparse.ArgumentParser(description="Trains and tests a given dataset and save the complete model for a Recurrent Neural Network.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_model_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--output_test_table_path', required=False, help='Path to the test table file. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Loss, accuracy and MSE plots. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    RecurrentNeuralNetwork(input_dataset_path=args.input_dataset_path,
                   output_model_path=args.output_model_path, 
                   output_test_table_path=args.output_test_table_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()
