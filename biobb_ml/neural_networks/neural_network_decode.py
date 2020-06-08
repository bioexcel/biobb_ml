#!/usr/bin/env python3

"""Module containing the DecodingNeuralNetwork class and the command line interface."""
import argparse
import h5py
import json
from tensorflow.python.keras.saving import hdf5_format
from sklearn.preprocessing import scale
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.neural_networks.common import *

class DecodingNeuralNetwork():
    """Decodes and predicts given a dataset and a model file.
    Wrapper of the TensorFlow Keras Long Short-Term Memory layer
    Visit the 'TensorFlow official website <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>'_. 

    Args:
        input_decode_path (str): Path to the input decode dataset. Accepted formats: csv.
        input_model_path (str): Path to the input model. Accepted formats: csv.
        output_decode_path (str): Path to the output decode file. Accepted formats: csv.
        output_predict_path (str): Path to the output predict file. Accepted formats: csv.
        properties (dic):
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_decode_path, input_model_path, 
                 output_decode_path, output_predict_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_decode_path": input_decode_path, "input_model_path": input_model_path }, 
            "out": { "output_decode_path": output_decode_path, "output_predict_path": output_predict_path } 
        }

        # Properties specific for BB
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
        self.io_dict["in"]["input_model_path"] = check_input_path(self.io_dict["in"]["input_model_path"], "input_model_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_decode_path"] = check_output_path(self.io_dict["out"]["output_decode_path"],"output_decode_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_predict_path"] = check_output_path(self.io_dict["out"]["output_predict_path"],"output_predict_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the DecodingNeuralNetwork module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_decode_path"], self.io_dict["out"]["output_predict_path"]]
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

        fu.log('Getting model from %s' % self.io_dict["in"]["input_model_path"], out_log, self.global_log)
        with h5py.File(self.io_dict["in"]["input_model_path"], mode='r') as f:
            variables = f.attrs['variables']
            new_model = hdf5_format.load_model_from_hdf5(f)

        # get dictionary with variables
        vars_obj = json.loads(variables)

        stringlist = []
        new_model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        fu.log('Model summary:\n\n%s\n' % model_summary, out_log, self.global_log)

        # decoding / predicting
        fu.log('Decoding / Predicting model', out_log, self.global_log)
        yhat = new_model.predict(seq_in, verbose = 1)

        # decoding
        decoding_table = pd.DataFrame()
        decoding_table['reconstructed'] = np.squeeze(np.asarray(yhat[0][0]))

        pd.set_option('display.float_format', lambda x: '%.5f' % x)
        decoding_table = decoding_table.reset_index(drop=True)
        fu.log('RECONSTRUCTION TABLE\n\n%s\n' % decoding_table, out_log, self.global_log)

        fu.log('Saving reconstruction to %s' % self.io_dict["out"]["output_decode_path"], out_log, self.global_log)
        decoding_table.to_csv(self.io_dict["out"]["output_decode_path"], index = False, header=True, float_format='%.5f')

        if len(yhat) == 2:
            # decoding
            prediction_table = pd.DataFrame()
            prediction_table['predicted'] = np.squeeze(np.asarray(yhat[1][0]))

            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            prediction_table = prediction_table.reset_index(drop=True)
            fu.log('PREDICTION TABLE\n\n%s\n' % prediction_table, out_log, self.global_log)

            fu.log('Saving prediction to %s' % self.io_dict["out"]["output_predict_path"], out_log, self.global_log)
            prediction_table.to_csv(self.io_dict["out"]["output_predict_path"], index = False, header=True, float_format='%.5f')

        return 0

def main():
    parser = argparse.ArgumentParser(description="Decodes and predicts given a dataset and a model file.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_decode_path', required=True, help='Path to the input decode dataset. Accepted formats: csv.')
    required_args.add_argument('--input_model_path', required=True, help='Path to the input model. Accepted formats: csv.')
    required_args.add_argument('--output_decode_path', required=True, help='Path to the output decode file. Accepted formats: csv.')
    parser.add_argument('--output_predict_path', required=False, help='Path to the output predict file. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    DecodingNeuralNetwork(input_decode_path=args.input_decode_path,
                   input_model_path=args.input_model_path,
                   output_decode_path=args.output_decode_path, 
                   output_predict_path=args.output_predict_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()
