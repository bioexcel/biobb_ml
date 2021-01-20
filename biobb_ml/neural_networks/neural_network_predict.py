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
    """
    | biobb_ml PredictNeuralNetwork
    | Makes predictions from an input dataset and a given model.
    | Makes predictions from an input dataset (provided either as a file or as a dictionary property) and a given model trained with `TensorFlow Keras Sequential <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`_ and `TensorFlow Keras LSTM <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM>`_

    Args:
        input_model_path (str): Path to the input model. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/input_model_predict.h5>`_. Accepted formats: h5 (edam:format_3590).
        input_dataset_path (str) (Optional): Path to the dataset to predict. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_predict.csv>`_. Accepted formats: csv (edam:format_3752).
        output_results_path (str): Path to the output results file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_predict.csv>`_. Accepted formats: csv (edam:format_3752).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets. It will be taken into account only in case **input_dataset_path** is not provided. Format: [{ 'var1': 1.0, 'var2': 2.0 }, { 'var1': 4.0, 'var2': 2.7 }] for datasets with headers and [[ 1.0, 2.0 ], [ 4.0, 2.7 ]] for datasets without headers.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.neural_networks.neural_network_predict import neural_network_predict
            prop = { 
                'predictions': [
                    { 
                        'var1': 1.0, 
                        'var2': 2.0 
                    }, 
                    { 
                        'var1': 4.0, 
                        'var2': 2.7 
                    }
                ] 
            }
            neural_network_predict(input_model_path='/path/to/myModel.h5', 
                                    input_dataset_path='/path/to/myDataset.csv', 
                                    output_results_path='/path/to/newPredictedResults.csv',
                                    properties=prop)

    Info:
        * wrapped_software:
            * name: TensorFlow
            * version: >2.1.0
            * license: MIT
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl
            
    """

    def __init__(self, input_model_path, output_results_path, 
                input_dataset_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_model_path": input_model_path, "input_dataset_path": input_dataset_path }, 
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
        if self.io_dict["in"]["input_dataset_path"]:
            self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`PredictNeuralNetwork <neural_networks.neural_network_predict.PredictNeuralNetwork>` neural_networks.neural_network_predict.PredictNeuralNetwork object."""

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

        if self.io_dict["in"]["input_dataset_path"]:
            # load dataset from input_dataset_path file
            fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
            if 'features' not in vars_obj:
                # recurrent
                labels = None
                skiprows = None
                with open(self.io_dict["in"]["input_dataset_path"]) as csvfile:
                    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
                    for row in reader: # each row is a list
                        self.predictions.append(row)
            else:
                # classification or regression
                if 'columns' in vars_obj['features']:
                    labels = getHeader(self.io_dict["in"]["input_dataset_path"])
                    skiprows = 1
                else:
                    labels = None
                    skiprows = None
            new_data_table = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = None, sep="\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)
        else:
            if vars_obj['type'] != 'recurrent':
                new_data_table = pd.DataFrame(data=get_list_of_predictors(self.predictions),columns=get_keys_of_predictors(self.predictions))
            else:
                new_data_table = pd.DataFrame(data=self.predictions, columns=get_num_cols(vars_obj['window_size']))

        # prediction
        if vars_obj['type'] != 'recurrent':
            # classification or regression

            #new_data_table = pd.DataFrame(data=get_list_of_predictors(self.predictions),columns=get_keys_of_predictors(self.predictions))
            new_data = new_data_table
            if vars_obj['scale']: new_data = scale(new_data)

            predictions = new_model.predict(new_data)
            predictions = np.around(predictions, decimals=2)

            clss = ''
            #if predictions.shape[1] > 1:
            if vars_obj['type'] == 'classification':
                # classification
                pr = tuple(map(tuple, predictions))
                clss = ' (' + ', '.join(str(x) for x in vars_obj['vs']) + ')'
            else:
                # regression
                pr = np.squeeze(np.asarray(predictions))

            new_data_table[getTargetValue(vars_obj['target']) + clss] = pr

        else:
            # recurrent

            #new_data_table = pd.DataFrame(data=self.predictions, columns=get_num_cols(vars_obj['window_size']))
            predictions = []

            for r in self.predictions:
                row = np.asarray(r).reshape((1, vars_obj['window_size'], 1))

                pred = new_model.predict(row)
                pred = np.around(pred, decimals=2)

                predictions.append(pred[0][0])

            #pd.set_option('display.float_format', lambda x: '%.2f' % x)
            new_data_table["predictions"] = predictions 

        fu.log('Predicting results\n\nPREDICTION RESULTS\n\n%s\n' % new_data_table, out_log, self.global_log)
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        new_data_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        return 0

def neural_network_predict(input_model_path: str, output_results_path: str, input_dataset_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`PredictNeuralNetwork <neural_networks.neural_network_predict.PredictNeuralNetwork>` class and
    execute the :meth:`launch() <neural_networks.neural_network_predict.PredictNeuralNetwork.launch>` method."""

    return PredictNeuralNetwork(input_model_path=input_model_path,  
                   output_results_path=output_results_path, 
                   input_dataset_path=input_dataset_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Makes predictions from an input dataset and a given classification model.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_model_path', required=True, help='Path to the input model. Accepted formats: h5.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--input_dataset_path', required=False, help='Path to the dataset to predict. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    neural_network_predict(input_model_path=args.input_model_path,
                           output_results_path=args.output_results_path, 
                           input_dataset_path=args.input_dataset_path,
                           properties=properties)

if __name__ == '__main__':
    main()
