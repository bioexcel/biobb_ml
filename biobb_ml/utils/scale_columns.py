#!/usr/bin/env python3

"""Module containing the ScaleColumns class and the command line interface."""
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.utils.common import *

class ScaleColumns():
    """Scales columns from a given dataset

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <>`_. Accepted formats: csv.
        output_dataset_path (str): Path to the output dataset. File type: output. `Sample file <>`_. Accepted formats: csv.
        properties (dic):
            * **columns** (*list*) - (None)  List of columns to drop from input dataset.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path, 
                 output_dataset_path, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_dataset_path": output_dataset_path } 
        }

        # Properties specific for BB
        self.columns = properties.get('columns', [])
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
        self.io_dict["out"]["output_dataset_path"] = check_output_path(self.io_dict["out"]["output_dataset_path"],"output_dataset_path", False, out_log, self.__class__.__name__)


    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the ScaleColumns module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_dataset_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        if all(isinstance(n, str) for n in self.columns):
            header = 0
        else:
            header = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = header, sep="\s+|;|:|,|\t", engine="python")

        fu.log('Scaling [%s] columns from dataset' % ', '.join(self.columns), out_log, self.global_log)
        df_scaled = (data[self.columns])

        scaler = MinMaxScaler()

        df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled))

        data[self.columns] = df_scaled

        hdr = False
        if header == 0: hdr = True
        fu.log('Saving dataset to %s' % self.io_dict["out"]["output_dataset_path"], out_log, self.global_log)
        data.to_csv(self.io_dict["out"]["output_dataset_path"], index = False, header=hdr)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Scales columns from a given dataset", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_dataset_path', required=True, help='Path to the output dataset. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    ScaleColumns(input_dataset_path=args.input_dataset_path,
                   output_dataset_path=args.output_dataset_path,
                   properties=properties).launch()

if __name__ == '__main__':
    main()

