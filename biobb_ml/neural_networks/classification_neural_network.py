#!/usr/bin/env python3

"""Module containing the ClassificationNeuralNetwork class and the command line interface."""
import argparse
import h5py
import json
from tensorflow.python.keras.saving import hdf5_format
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import math
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.neural_networks.common import *


class ClassificationNeuralNetwork():
    """
    | biobb_ml ClassificationNeuralNetwork
    | Wrapper of the TensorFlow Keras Sequential method for classification. 
    | Trains and tests a given dataset and save the complete model for a Neural Network Classification. Visit the `Sequential documentation page <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`_ in the TensorFlow Keras official website for further information. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_classification.csv>`_. Accepted formats: csv (edam:format_3752).
        output_model_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_model_classification.h5>`_. Accepted formats: h5 (edam:format_3590).
        output_test_table_path (str) (Optional): Path to the test table file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_classification.csv>`_. Accepted formats: csv (edam:format_3752).
        output_plot_path (str) (Optional): Loss, accuracy and MSE plots. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_plot_classification.png>`_. Accepted formats: png (edam:format_3603).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **features** (*dict*) - ({}) Independent variables or columns from your dataset you want to train. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **target** (*dict*) - ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **weight** (*dict*) - ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of multiple formats, the first one will be picked.
            * **validation_size** (*float*) - (0.2) [0~1|0.05] Represents the proportion of the dataset to include in the validation split. It should be between 0.0 and 1.0.
            * **test_size** (*float*) - (0.1) [0~1|0.05] Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
            * **hidden_layers** (*list*) - (None)  List of dictionaries with hidden layers values. Format: [ { 'size': 50, 'activation': 'relu' } ].
            * **output_layer_activation** (*string*) - ("softmax") Activation function to use in the output layer. Values: sigmoid (Sigmoid activation function: sigmoid[x] = 1 / [1 + exp[-x]]), tanh (Hyperbolic tangent activation function), relu (Applies the rectified linear unit activation function), softmax (Softmax converts a real vector to a vector of categorical probabilities).
            * **optimizer** (*string*) - ("Adam") Name of optimizer instance. Values: Adadelta (Adadelta optimization is a stochastic gradient descent method that is based on adaptive learning rate per dimension to address two drawbacks: the continual decay of learning rates throughout training and the need for a manually selected global learning rate), Adagrad (Adagrad is an optimizer with parameter-specific learning rates; which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives; the smaller the updates), Adam (Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments), Adamax (It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper. Adamax is sometimes superior to adam; specially in models with embeddings), Ftrl (Optimizer that implements the FTRL algorithm), Nadam (Much like Adam is essentially RMSprop with momentum; Nadam is Adam with Nesterov momentum), RMSprop (Optimizer that implements the RMSprop algorithm), SGD (Gradient descent -with momentum- optimizer).
            * **learning_rate** (*float*) - (0.02) [0~100|0.01] Determines the step size at each iteration while moving toward a minimum of a loss function
            * **batch_size** (*int*) - (100) [0~1000|1] Number of samples per gradient update.
            * **max_epochs** (*int*) - (100) [0~1000|1] Number of epochs to train the model. As the early stopping is enabled, this is a maximum.
            * **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
            * **random_state** (*int*) - (5) [1~1000|1] Controls the shuffling applied to the data before applying the split. .
            * **scale** (*bool*) - (False) Whether or not to scale the input dataset.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.neural_networks.classification_neural_network import classification_neural_network
            prop = { 
                'features': {
                    'columns': [ 'column1', 'column2', 'column3' ] 
                },
                'target': { 
                    'column': 'target' 
                },
                'validation_size': 0.2,
                'test_size': .33,
                'hidden_layers': [
                    { 
                        'size': 10, 
                        'activation': 'relu' 
                    },
                    { 
                        'size': 8, 
                        'activation': 'relu' 
                    }
                ],
                'optimizer': 'Adam',
                'learning_rate': 0.01,
                'batch_size': 32,
                'max_epochs': 150
            }
            classification_neural_network(input_dataset_path='/path/to/myDataset.csv', 
                                        output_model_path='/path/to/newModel.h5', 
                                        output_test_table_path='/path/to/newTable.csv', 
                                        output_plot_path='/path/to/newPlot.png',
                                        properties=prop)

    Info:
        * wrapped_software:
            * name: TensorFlow Keras Sequential
            * version: >2.1.0
            * license: MIT
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
        self.features = properties.get('features', {})
        self.target = properties.get('target', {})
        self.weight = properties.get('weight', {})
        self.validation_size = properties.get('validation_size', 0.1)
        self.test_size = properties.get('test_size', 0.1)
        self.hidden_layers = properties.get('hidden_layers', [])
        self.output_layer_activation = properties.get('output_layer_activation', 'softmax')
        self.optimizer = properties.get('optimizer', 'Adam')
        self.learning_rate = properties.get('learning_rate', 0.02)
        self.batch_size = properties.get('batch_size', 100)
        self.max_epochs = properties.get('max_epochs', 100)
        self.normalize_cm =  properties.get('normalize_cm', False)
        self.random_state = properties.get('random_state', 5)
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
        self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_model_path"] = check_output_path(self.io_dict["out"]["output_model_path"],"output_model_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_test_table_path"] = check_output_path(self.io_dict["out"]["output_test_table_path"],"output_test_table_path", True, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    def build_model(self, input_shape, output_size):
        """ Builds Neural network according to hidden_layers property """

        # create model
        model = Sequential([])

        # if no hidden_layers provided, create manually a hidden layer with default values
        if not self.hidden_layers:
            self.hidden_layers = [ { 'size': 50, 'activation': 'relu' } ]

        # generate hidden_layers
        for i,layer in enumerate(self.hidden_layers):
            if i == 0:
                model.add(Dense(layer['size'], activation = layer['activation'], kernel_initializer='he_normal', input_shape = input_shape)) # 1st hidden layer
            else:
                model.add(Dense(layer['size'], activation = layer['activation'], kernel_initializer='he_normal'))

        model.add(Dense(output_size, activation=self.output_layer_activation)) # output layer

        return model

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`ClassificationNeuralNetwork <neural_networks.classification_neural_network.ClassificationNeuralNetwork>` neural_networks.classification_neural_network.ClassificationNeuralNetwork object."""

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
        if 'columns' in self.features:
            labels = getHeader(self.io_dict["in"]["input_dataset_path"])
            skiprows = 1
        else:
            labels = None
            skiprows = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = None, sep="\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)

        targets_list = data[getTargetValue(self.target)].to_numpy()

        X = getFeatures(self.features, data, out_log, self.__class__.__name__)
        fu.log('Features: [%s]' % (getIndependentVarsList(self.features)), out_log, self.global_log)
        # target
        y = getTarget(self.target, data, out_log, self.__class__.__name__)
        fu.log('Target: %s' % (str(getTargetValue(self.target))), out_log, self.global_log)
        # weights
        if self.weight:
            w = getWeight(self.weight, data, out_log, self.__class__.__name__)

        # shuffle dataset
        fu.log('Shuffling dataset', out_log, self.global_log)
        shuffled_indices = np.arange(X.shape[0])
        np.random.shuffle(shuffled_indices)
        np_X = X.to_numpy()
        shuffled_X = np_X[shuffled_indices]
        shuffled_y = targets_list[shuffled_indices]
        if self.weight: shuffled_w = w[shuffled_indices]

        # train / test split
        fu.log('Creating train and test sets', out_log, self.global_log)
        arrays_sets = (shuffled_X, shuffled_y)
        # if user provide weights
        if self.weight:
            arrays_sets = arrays_sets + (shuffled_w,)
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(*arrays_sets, test_size=self.test_size, random_state = self.random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(*arrays_sets, test_size=self.test_size, random_state = self.random_state)

        # scale dataset
        if self.scale: 
            fu.log('Scaling dataset', out_log, self.global_log)
            X_train = scale(X_train)

        # build model
        fu.log('Building model', out_log, self.global_log)
        model = self.build_model((X_train.shape[1],), np.unique(y_train).size)

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
        model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy', 'mse'])

        # fitting
        fu.log('Training model', out_log, self.global_log)
        # set an early stopping mechanism
        # set patience=2, to be a bit tolerant against random validation loss increases
        early_stopping = EarlyStopping(patience=2)

        if self.weight: 
            sample_weight = w_train
            class_weight = []
        else:
            # TODO: class_weight not working since TF 2.4.1 update
            #fu.log('No weight provided, class_weight will be estimated from the target data', out_log, self.global_log)
            fu.log('No weight provided', out_log, self.global_log)
            sample_weight = None
            class_weight = []#compute_class_weight('balanced', np.unique(y_train), y_train)

        print(class_weight)
        # fit the model
        mf = model.fit(X_train, 
                       y_train, 
                       class_weight=class_weight,
                       sample_weight = sample_weight,
                       batch_size=self.batch_size, 
                       epochs=self.max_epochs, 
                       callbacks=[early_stopping],
                       validation_split=self.validation_size,
                       verbose = 1)

        fu.log('Total epochs performed: %s' % len(mf.history['loss']), out_log, self.global_log)

        train_metrics = pd.DataFrame()
        train_metrics['metric'] = ['Train loss',' Train accuracy', 'Train MSE', 'Validation loss', 'Validation accuracy', 'Validation MSE']
        train_metrics['coefficient'] = [mf.history['loss'][-1], mf.history['accuracy'][-1], mf.history['mse'][-1], mf.history['val_loss'][-1], mf.history['val_accuracy'][-1], mf.history['val_mse'][-1]]

        fu.log('Training metrics\n\nTRAINING METRICS TABLE\n\n%s\n' % train_metrics, out_log, self.global_log)

        # confusion matrix
        train_predictions = model.predict(X_train)
        train_predictions = np.around(train_predictions, decimals=2)
        norm_pred = []
        [norm_pred.append(np.argmax(pred, axis=0)) for pred in train_predictions]
        cnf_matrix_train = math.confusion_matrix(y_train, norm_pred).numpy()
        np.set_printoptions(precision=2)
        if self.normalize_cm:
            cnf_matrix_train = cnf_matrix_train.astype('float') / cnf_matrix_train.sum(axis=1)[:, np.newaxis]
            cm_type = 'NORMALIZED CONFUSION MATRIX'
        else:
            cm_type = 'CONFUSION MATRIX, WITHOUT NORMALIZATION'

        fu.log('Calculating confusion matrix for training dataset\n\n%s\n\n%s\n' % (cm_type, cnf_matrix_train), out_log, self.global_log)

        # testing
        if self.scale: 
            X_test = scale(X_test)
        fu.log('Testing model', out_log, self.global_log)
        test_loss, test_accuracy, test_mse = model.evaluate(X_test, y_test)

        test_metrics = pd.DataFrame()
        test_metrics['metric'] = ['Test loss',' Test accuracy', 'Test MSE']
        test_metrics['coefficient'] = [test_loss, test_accuracy, test_mse]

        fu.log('Testing metrics\n\nTESTING METRICS TABLE\n\n%s\n' % test_metrics, out_log, self.global_log)

        # predict data from X_test
        test_predictions = model.predict(X_test)
        test_predictions = np.around(test_predictions, decimals=2)
        tpr = tuple(map(tuple, test_predictions))

        test_table = pd.DataFrame()
        test_table['P' + np.array2string(np.unique(y_train))] = tpr
        test_table['target'] = y_test

        fu.log('TEST DATA\n\n%s\n' % test_table, out_log, self.global_log)

        # confusion matrix
        norm_pred = []
        [norm_pred.append(np.argmax(pred, axis=0)) for pred in test_predictions]
        cnf_matrix_test = math.confusion_matrix(y_test, norm_pred).numpy()
        np.set_printoptions(precision=2)
        if self.normalize_cm:
            cnf_matrix_test = cnf_matrix_test.astype('float') / cnf_matrix_test.sum(axis=1)[:, np.newaxis]
            cm_type = 'NORMALIZED CONFUSION MATRIX'
        else:
            cm_type = 'CONFUSION MATRIX, WITHOUT NORMALIZATION'

        fu.log('Calculating confusion matrix for testing dataset\n\n%s\n\n%s\n' % (cm_type, cnf_matrix_test), out_log, self.global_log)

        # save test data
        if(self.io_dict["out"]["output_test_table_path"]): 
            fu.log('Saving testing data to %s' % self.io_dict["out"]["output_test_table_path"], out_log, self.global_log)
            test_table.to_csv(self.io_dict["out"]["output_test_table_path"], index = False, header=True)

        # create test plot
        if(self.io_dict["out"]["output_plot_path"]): 
            vs = np.unique(targets_list)
            vs.sort()
            if len(vs) > 2:
                plot = plotResultsClassMultCM(mf.history, cnf_matrix_train, cnf_matrix_test, self.normalize_cm, vs)
                fu.log('Saving confusion matrix plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            else:
                plot = plotResultsClassBinCM(mf.history, train_predictions, test_predictions, y_train, y_test, cnf_matrix_train, cnf_matrix_test, self.normalize_cm, vs)
                fu.log('Saving binary classifier evaluator plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        # save model and parameters
        vs = np.unique(targets_list)
        vs.sort()
        vars_obj = {
            'features': self.features,
            'target': self.target,
            'scale': self.scale,
            'vs': vs.tolist(),
            'type': 'classification'
        }
        variables = json.dumps(vars_obj)
        fu.log('Saving model to %s' % self.io_dict["out"]["output_model_path"], out_log, self.global_log)
        with h5py.File(self.io_dict["out"]["output_model_path"], mode='w') as f:
            hdf5_format.save_model_to_hdf5(model, f)
            f.attrs['variables'] = variables

        return 0

def classification_neural_network(input_dataset_path: str, output_model_path: str, output_test_table_path: str = None, output_plot_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`AutoencoderNeuralNetwork <neural_networks.classification_neural_network.AutoencoderNeuralNetwork>` class and
    execute the :meth:`launch() <neural_networks.classification_neural_network.AutoencoderNeuralNetwork.launch>` method."""

    return ClassificationNeuralNetwork(input_dataset_path=input_dataset_path,  
                   output_model_path=output_model_path, 
                   output_test_table_path=output_test_table_path,
                   output_plot_path=output_plot_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of the TensorFlow Keras Sequential method.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_model_path', required=True, help='Path to the output model file. Accepted formats: h5.')
    parser.add_argument('--output_test_table_path', required=False, help='Path to the test table file. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Loss, accuracy and MSE plots. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    classification_neural_network(input_dataset_path=args.input_dataset_path,
                                   output_model_path=args.output_model_path, 
                                   output_test_table_path=args.output_test_table_path, 
                                   output_plot_path=args.output_plot_path, 
                                   properties=properties)

if __name__ == '__main__':
    main()
