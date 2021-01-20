#!/usr/bin/env python3

"""Module containing the AutoencoderNeuralNetwork class and the command line interface."""
import argparse
import h5py
import json
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.neural_networks.common import *


class AutoencoderNeuralNetwork():
    """
    | biobb_ml AutoencoderNeuralNetwork
    | Wrapper of the TensorFlow Keras LSTM method for encoding. 
    | Fits and tests a given dataset and save the compiled model for an Autoencoder Neural Network. Visit the `LSTM documentation page <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>`_ in the TensorFlow Keras official website for further information. 

    Args:
        input_decode_path (str): Path to the input decode dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_autoencoder_decode.csv>`_. Accepted formats: csv (edam:format_3752).
        input_predict_path (str) (Optional): Path to the input predict dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_autoencoder_predict.csv>`_. Accepted formats: csv (edam:format_3752).
        output_model_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_model_autoencoder.h5>`_. Accepted formats: h5 (edam:format_3590).
        output_test_decode_path (str) (Optional): Path to the test decode table file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_decode_autoencoder.csv>`_. Accepted formats: csv (edam:format_3752).
        output_test_predict_path (str) (Optional): Path to the test predict table file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_predict_autoencoder.csv>`_. Accepted formats: csv (edam:format_3752).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **optimizer** (*string*) - ("Adam") Name of optimizer instance. Values: Adadelta (Adadelta optimization is a stochastic gradient descent method that is based on adaptive learning rate per dimension to address two drawbacks: the continual decay of learning rates throughout training and the need for a manually selected global learning rate), Adagrad (Adagrad is an optimizer with parameter-specific learning rates; which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives; the smaller the updates), Adam (Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments), Adamax (It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper. Adamax is sometimes superior to adam; specially in models with embeddings), Ftrl (Optimizer that implements the FTRL algorithm), Nadam (Much like Adam is essentially RMSprop with momentum; Nadam is Adam with Nesterov momentum), RMSprop (Optimizer that implements the RMSprop algorithm), SGD (Gradient descent -with momentum- optimizer).
            * **learning_rate** (*float*) - (0.02) [0~100|0.01] Determines the step size at each iteration while moving toward a minimum of a loss function
            * **batch_size** (*int*) - (100) [0~1000|1] Number of samples per gradient update.
            * **max_epochs** (*int*) - (100) [0~1000|1] Number of epochs to train the model. As the early stopping is enabled, this is a maximum.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.neural_networks.autoencoder_neural_network import autoencoder_neural_network
            prop = { 
                'optimizer': 'Adam', 
                'learning_rate': 0.01, 
                'batch_size': 32,
                'max_epochs': 300
            }
            autoencoder_neural_network(input_decode_path='/path/to/myDecodeDataset.csv', 
                                        output_model_path='/path/to/newModel.h5', 
                                        input_predict_path='/path/to/myPredictDataset.csv', 
                                        output_test_decode_path='/path/to/newDecodeDataset.csv', 
                                        output_test_predict_path='/path/to/newPredictDataset.csv', 
                                        properties=prop)

    Info:
        * wrapped_software:
            * name: TensorFlow Keras LSTM
            * version: >2.1.0
            * license: MIT
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_decode_path, output_model_path, 
                input_predict_path=None, output_test_decode_path=None, output_test_predict_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_decode_path": input_decode_path, "input_predict_path": input_predict_path }, 
            "out": { "output_model_path": output_model_path, "output_test_decode_path": output_test_decode_path, "output_test_predict_path": output_test_predict_path } 
        }

        # Properties specific for BB
        self.optimizer = properties.get('optimizer', 'Adam')
        self.learning_rate = properties.get('learning_rate', 0.02)
        self.batch_size = properties.get('batch_size', 100)
        self.max_epochs = properties.get('max_epochs', 100)
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
        self.io_dict["in"]["input_decode_path"] = check_input_path(self.io_dict["in"]["input_decode_path"], "input_decode_path", False, out_log, self.__class__.__name__)
        if self.io_dict["in"]["input_predict_path"]:
            self.io_dict["in"]["input_predict_path"] = check_input_path(self.io_dict["in"]["input_predict_path"], "input_predict_path", True, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_model_path"] = check_output_path(self.io_dict["out"]["output_model_path"],"output_model_path", False, out_log, self.__class__.__name__)
        if self.io_dict["out"]["output_test_decode_path"]:
            self.io_dict["out"]["output_test_decode_path"] = check_output_path(self.io_dict["out"]["output_test_decode_path"],"output_test_decode_path", True, out_log, self.__class__.__name__)
        if self.io_dict["out"]["output_test_predict_path"]:
            self.io_dict["out"]["output_test_predict_path"] = check_output_path(self.io_dict["out"]["output_test_predict_path"],"output_test_predict_path", True, out_log, self.__class__.__name__)

    def build_model(self, n_in, n_out = None):
        """ Builds Neural network """

        # outputs list
        outputs = []

        # define encoder
        visible = Input(shape=(n_in,1))
        encoder = LSTM(100, activation='relu')(visible)

        # define reconstruct decoder
        decoder1 = RepeatVector(n_in)(encoder)
        decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
        decoder1 = TimeDistributed(Dense(1))(decoder1)

        outputs.append(decoder1)

        # define predict decoder
        if n_out:
            decoder2 = RepeatVector(n_out)(encoder)
            decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
            decoder2 = TimeDistributed(Dense(1))(decoder2)
            outputs.append(decoder2)

        # tie it together
        model = Model(inputs=visible, outputs=outputs)

        return model

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`AutoencoderNeuralNetwork <neural_networks.autoencoder_neural_network.AutoencoderNeuralNetwork>` neural_networks.autoencoder_neural_network.AutoencoderNeuralNetwork object."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_model_path"],self.io_dict["out"]["output_test_decode_path"],self.io_dict["out"]["output_test_predict_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load decode dataset
        fu.log('Getting decode dataset from %s' % self.io_dict["in"]["input_decode_path"], out_log, self.global_log)
        data_dec = pd.read_csv(self.io_dict["in"]["input_decode_path"])
        seq_in = np.array(data_dec)

        # reshape input into [samples, timesteps, features]
        n_in = len(seq_in)
        seq_in = seq_in.reshape((1, n_in, 1))

        # load predict dataset
        n_out = None
        if(self.io_dict["in"]["input_predict_path"]): 
            fu.log('Getting predict dataset from %s' % self.io_dict["in"]["input_predict_path"], out_log, self.global_log)
            data_pred = pd.read_csv(self.io_dict["in"]["input_predict_path"])
            seq_out = np.array(data_pred)

            # reshape output into [samples, timesteps, features]
            n_out = len(seq_out)
            seq_out = seq_out.reshape((1, n_out, 1))

        # build model
        fu.log('Building model', out_log, self.global_log)
        model = self.build_model(n_in, n_out)

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
        y_list = [seq_in]
        if n_out:
            y_list.append(seq_out)
        # fit the model
        mf = model.fit(seq_in, 
                       y_list, 
                       batch_size=self.batch_size, 
                       epochs=self.max_epochs, 
                       verbose = 1)

        train_metrics = pd.DataFrame()
        metric = []
        coefficient = []
        for key, lst in mf.history.items():
            metric.append(' '.join(x.capitalize() or '_' for x in key.split('_')))
            coefficient.append(lst[-1])

        train_metrics['metric'] = metric
        train_metrics['coefficient'] = coefficient

        fu.log('Calculating metrics\n\nMETRICS TABLE\n\n%s\n' % train_metrics, out_log, self.global_log)

        # predicting
        fu.log('Predicting model', out_log, self.global_log)
        yhat = model.predict(seq_in, verbose=1)

        decoding_table = pd.DataFrame()
        if(self.io_dict["in"]["input_predict_path"]): 
            decoding_table['reconstructed'] = np.squeeze(np.asarray(yhat[0][0]))
            decoding_table['original'] = data_dec
        else:
            decoding_table['reconstructed'] = np.squeeze(np.asarray(yhat[0]))
            decoding_table['original'] = np.squeeze(np.asarray(data_dec))
        decoding_table['residual'] = decoding_table['original'] - decoding_table['reconstructed']
        decoding_table['difference %'] = np.absolute(decoding_table['residual']/decoding_table['original']*100)
        pd.set_option('display.float_format', lambda x: '%.5f' % x)
        # sort by difference in %
        decoding_table = decoding_table.sort_values(by=['difference %'])
        decoding_table = decoding_table.reset_index(drop=True)
        fu.log('RECONSTRUCTION TABLE\n\n%s\n' % decoding_table, out_log, self.global_log)

        # save reconstruction data
        if(self.io_dict["out"]["output_test_decode_path"]): 
            fu.log('Saving reconstruction data to %s' % self.io_dict["out"]["output_test_decode_path"], out_log, self.global_log)
            decoding_table.to_csv(self.io_dict["out"]["output_test_decode_path"], index = False, header=True)

        if(self.io_dict["in"]["input_predict_path"]): 
            prediction_table = pd.DataFrame()
            prediction_table['predicted'] = np.squeeze(np.asarray(yhat[1][0]))
            prediction_table['original'] = data_pred
            prediction_table['residual'] = prediction_table['original'] - prediction_table['predicted']
            prediction_table['difference %'] = np.absolute(prediction_table['residual']/prediction_table['original']*100)
            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            # sort by difference in %
            prediction_table = prediction_table.sort_values(by=['difference %'])
            prediction_table = prediction_table.reset_index(drop=True)
            fu.log('PREDICTION TABLE\n\n%s\n' % prediction_table, out_log, self.global_log)

            # save decoding data
            if(self.io_dict["out"]["output_test_predict_path"]): 
                fu.log('Saving prediction data to %s' % self.io_dict["out"]["output_test_predict_path"], out_log, self.global_log)
                prediction_table.to_csv(self.io_dict["out"]["output_test_predict_path"], index = False, header=True)

        # save model and parameters
        vars_obj = {
            'type': 'autoencoder'
        }
        variables = json.dumps(vars_obj)
        fu.log('Saving model to %s' % self.io_dict["out"]["output_model_path"], out_log, self.global_log)
        with h5py.File(self.io_dict["out"]["output_model_path"], mode='w') as f:
            hdf5_format.save_model_to_hdf5(model, f)
            f.attrs['variables'] = variables

        return 0

def autoencoder_neural_network(input_decode_path: str, output_model_path: str, input_predict_path: str = None, output_test_decode_path: str = None, output_test_predict_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`AutoencoderNeuralNetwork <neural_networks.autoencoder_neural_network.AutoencoderNeuralNetwork>` class and
    execute the :meth:`launch() <neural_networks.autoencoder_neural_network.AutoencoderNeuralNetwork.launch>` method."""

    return AutoencoderNeuralNetwork(input_decode_path=input_decode_path,  
                    output_model_path=output_model_path, 
                    input_predict_path=input_predict_path, 
                    output_test_decode_path=output_test_decode_path,
                    output_test_predict_path=output_test_predict_path,
                    properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of the TensorFlow Keras LSTM method for encoding.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_decode_path', required=True, help='Path to the input decode dataset. Accepted formats: csv.')
    parser.add_argument('--input_predict_path', required=False, help='Path to the input predict dataset. Accepted formats: csv.')
    required_args.add_argument('--output_model_path', required=True, help='Path to the output model file. Accepted formats: h5.')
    parser.add_argument('--output_test_decode_path', required=False, help='Path to the test decode table file. Accepted formats: csv.')
    parser.add_argument('--output_test_predict_path', required=False, help='Path to the test predict table file. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    autoencoder_neural_network(input_decode_path=args.input_decode_path,
                                output_model_path=args.output_model_path, 
                                input_predict_path=args.input_predict_path, 
                                output_test_decode_path=args.output_test_decode_path, 
                                output_test_predict_path=args.output_test_predict_path, 
                                properties=properties)

if __name__ == '__main__':
    main()
