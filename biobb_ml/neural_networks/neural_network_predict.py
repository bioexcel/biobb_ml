#!/usr/bin/env python3

"""Module containing the PredictNeuralNetwork class and the command line interface."""
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

class PredictNeuralNetwork():
    """Calculates prediction for a NN classification given a model file.
    Visit the `TensorFlow official website <https://www.tensorflow.org/api_docs/python/tf>`_. 

    Args:
        input_model_path (str): Path to the input model. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/input_model_predict.h5>`_. Accepted formats: h5.
        output_results_path (str): Path to the output results file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_predict.csv>`_. Accepted formats: csv.
        properties (dic):
            * **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_model_path,
                 output_results_path, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_model_path": input_model_path }, 
            "out": { "output_results_path": output_results_path } 
        }

        # Properties specific for BB
        self.predictions = properties.get('predictions', [])
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
        self.io_dict["in"]["input_model_path"] = check_input_path(self.io_dict["in"]["input_model_path"], "input_model_path", False, out_log, self.__class__.__name__)
        self.io_dict["out"]["output_results_path"] = check_output_path(self.io_dict["out"]["output_results_path"],"output_results_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the PredictNeuralNetwork module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_results_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

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

        # prediction
        if vars_obj['type'] != 'recurrent':
            # classification or regression

            new_data_table = pd.DataFrame(data=get_list_of_predictors(self.predictions),columns=get_keys_of_predictors(self.predictions))
            new_data = scale(new_data_table)

            predictions = new_model.predict(new_data)
            predictions = np.around(predictions, decimals=2)

            clss = ''
            if predictions.shape[1] > 1:
                # classification
                pr = tuple(map(tuple, predictions))
                clss = ' (' + ', '.join(str(x) for x in vars_obj['vs']) + ')'
            else:
                # regression
                pr = np.squeeze(np.asarray(predictions))
            new_data_table[vars_obj['target'] + clss] = pr

        else:
            # recurrent

            new_data_table = pd.DataFrame(data=self.predictions, columns=get_num_cols(vars_obj['window_size']))
            predictions = []

            for r in self.predictions:
                row = np.asarray(r).reshape((1, vars_obj['window_size'], 1))

                pred = new_model.predict(row)
                pred = np.around(pred, decimals=2)

                predictions.append(pred[0][0])

            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            new_data_table["predictions"] = predictions 
        
        fu.log('Predicting results\n\nPREDICTION RESULTS\n\n%s\n' % new_data_table, out_log, self.global_log)
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        new_data_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        return 0

def main():
    parser = argparse.ArgumentParser(description="Calculates prediction for a NN classification given a model file.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_model_path', required=True, help='Path to the input model. Accepted formats: h5.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    PredictNeuralNetwork(input_model_path=args.input_model_path,
                   output_results_path=args.output_results_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()

