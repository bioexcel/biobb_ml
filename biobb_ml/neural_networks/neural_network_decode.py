#!/usr/bin/env python3

"""Module containing the DecodingNeuralNetwork class and the command line interface."""
import argparse
import h5py
import json
from biobb_common.generic.biobb_object import BiobbObject
from tensorflow.python.keras.saving import hdf5_format
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_ml.neural_networks.common import *

class DecodingNeuralNetwork(BiobbObject):
    """
    | biobb_ml DecodingNeuralNetwork
    | Wrapper of the TensorFlow Keras LSTM method for decoding. 
    | Decodes and predicts given a dataset and a model file compiled by an Autoencoder Neural Network. Visit the `LSTM documentation page <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>`_ in the TensorFlow Keras official website for further information. 

    Args:
        input_decode_path (str): Path to the input decode dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_decoder.csv>`_. Accepted formats: csv (edam:format_3752).
        input_model_path (str): Path to the input model. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/input_model_decoder.h5>`_. Accepted formats: h5 (edam:format_3590).
        output_decode_path (str): Path to the output decode file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_decode_decoder.csv>`_. Accepted formats: csv (edam:format_3752).
        output_predict_path (str) (Optional): Path to the output predict file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_predict_decoder.csv>`_. Accepted formats: csv (edam:format_3752).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.neural_networks.neural_network_decode import neural_network_decode
            prop = { }
            neural_network_decode(input_decode_path='/path/to/myDecodeDataset.csv', 
                                input_model_path='/path/to/newModel.h5', 
                                output_decode_path='/path/to/newDecodeDataset.csv', 
                                output_predict_path='/path/to/newPredictDataset.csv', 
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

    def __init__(self, input_decode_path, input_model_path, output_decode_path, 
                output_predict_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Call parent class constructor
        super().__init__(properties)
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = { 
            "in": { "input_decode_path": input_decode_path, "input_model_path": input_model_path }, 
            "out": { "output_decode_path": output_decode_path, "output_predict_path": output_predict_path } 
        }

        # Properties specific for BB
        self.properties = properties

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

    def check_data_params(self, out_log, err_log):
        """ Checks all the input/output paths and parameters """
        self.io_dict["in"]["input_decode_path"] = check_input_path(self.io_dict["in"]["input_decode_path"], "input_decode_path", False, out_log, self.__class__.__name__)
        self.io_dict["in"]["input_model_path"] = check_input_path(self.io_dict["in"]["input_model_path"], "input_model_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_decode_path"] = check_output_path(self.io_dict["out"]["output_decode_path"],"output_decode_path", False, out_log, self.__class__.__name__)
        if self.io_dict["out"]["output_predict_path"]:
            self.io_dict["out"]["output_predict_path"] = check_output_path(self.io_dict["out"]["output_predict_path"],"output_predict_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`DecodingNeuralNetwork <neural_networks.neural_network_decode.DecodingNeuralNetwork>` neural_networks.neural_network_decode.DecodingNeuralNetwork object."""

        # check input/output paths and parameters
        self.check_data_params(self.out_log, self.err_log)

        # Setup Biobb
        if self.check_restart(): return 0
        self.stage_files()

        # load decode dataset
        fu.log('Getting decode dataset from %s' % self.io_dict["in"]["input_decode_path"], self.out_log, self.global_log)
        data_dec = pd.read_csv(self.io_dict["in"]["input_decode_path"])
        seq_in = np.array(data_dec)

        # reshape input into [samples, timesteps, features]
        n_in = len(seq_in)
        seq_in = seq_in.reshape((1, n_in, 1))

        fu.log('Getting model from %s' % self.io_dict["in"]["input_model_path"], self.out_log, self.global_log)
        with h5py.File(self.io_dict["in"]["input_model_path"], mode='r') as f:
            variables = f.attrs['variables']
            new_model = hdf5_format.load_model_from_hdf5(f)

        # get dictionary with variables
        vars_obj = json.loads(variables)

        stringlist = []
        new_model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        fu.log('Model summary:\n\n%s\n' % model_summary, self.out_log, self.global_log)

        # decoding / predicting
        fu.log('Decoding / Predicting model', self.out_log, self.global_log)
        yhat = new_model.predict(seq_in, verbose = 1)

        # decoding
        decoding_table = pd.DataFrame()
        decoding_table['reconstructed'] = np.squeeze(np.asarray(yhat[0][0]))

        pd.set_option('display.float_format', lambda x: '%.5f' % x)
        decoding_table = decoding_table.reset_index(drop=True)
        fu.log('RECONSTRUCTION TABLE\n\n%s\n' % decoding_table, self.out_log, self.global_log)

        fu.log('Saving reconstruction to %s' % self.io_dict["out"]["output_decode_path"], self.out_log, self.global_log)
        decoding_table.to_csv(self.io_dict["out"]["output_decode_path"], index = False, header=True, float_format='%.5f')

        if len(yhat) == 2:
            # decoding
            prediction_table = pd.DataFrame()
            prediction_table['predicted'] = np.squeeze(np.asarray(yhat[1][0]))

            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            prediction_table = prediction_table.reset_index(drop=True)
            fu.log('PREDICTION TABLE\n\n%s\n' % prediction_table, self.out_log, self.global_log)

            fu.log('Saving prediction to %s' % self.io_dict["out"]["output_predict_path"], self.out_log, self.global_log)
            prediction_table.to_csv(self.io_dict["out"]["output_predict_path"], index = False, header=True, float_format='%.5f')

        # Copy files to host
        self.copy_to_host()

        self.tmp_files.extend([
            self.stage_io_dict.get("unique_dir")
        ])
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)

        return 0

def neural_network_decode(input_decode_path: str, input_model_path: str, output_decode_path: str, output_predict_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`DecodingNeuralNetwork <neural_networks.neural_network_decode.DecodingNeuralNetwork>` class and
    execute the :meth:`launch() <neural_networks.neural_network_decode.DecodingNeuralNetwork.launch>` method."""

    return DecodingNeuralNetwork(input_decode_path=input_decode_path,  
                   input_model_path=input_model_path, 
                   output_decode_path=output_decode_path,
                   output_predict_path=output_predict_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of the TensorFlow Keras LSTM method for decoding.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_decode_path', required=True, help='Path to the input decode dataset. Accepted formats: csv.')
    required_args.add_argument('--input_model_path', required=True, help='Path to the input model. Accepted formats: h5.')
    required_args.add_argument('--output_decode_path', required=True, help='Path to the output decode file. Accepted formats: csv.')
    parser.add_argument('--output_predict_path', required=False, help='Path to the output predict file. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    neural_network_decode(input_decode_path=args.input_decode_path,
                           input_model_path=args.input_model_path,
                           output_decode_path=args.output_decode_path, 
                           output_predict_path=args.output_predict_path, 
                           properties=properties)

if __name__ == '__main__':
    main()
