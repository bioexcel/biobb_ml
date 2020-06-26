#!/usr/bin/env python3

"""Module containing the CorrelationMatrix class and the command line interface."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.utils.common import *


class CorrelationMatrix():
    """Generates a correlation matrix from a given dataset.

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_correlation_matrix.csv>`_. Accepted formats: csv.
        output_plot_path (str): Path to the correlation matrix plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_plot_correlation_matrix.png>`_. Accepted formats: png.
        properties (dic):
            * **features** (*list*) - (None) List with all features to compare.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_plot_path, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.features = properties.get('features', [])
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
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the CorrelationMatrix module."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_plot_path"]]
            if fu.check_complete_files(output_file_list):
                fu.log('Restart is enabled, this step: %s will the skipped' % self.step, out_log, self.global_log)
                return 0

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], out_log, self.global_log)
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])

        if self.features: data = data.filter(self.features)

        fu.log('Parsing dataset', out_log, self.global_log)
        if data.shape[1] < 10: 
            s = None
            fs = 12
        elif data.shape[1] >= 10 and data.shape[1] < 20: 
            s = (12,12)
            fs = 9
        elif data.shape[1] >= 20: 
            s = (16,16)
            fs = 7

        f, ax = plt.subplots(figsize=s)
        corr = data.corr()
        hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap='Blues', fmt='.2f', square=True, annot_kws={"fontsize":fs} )
        f.subplots_adjust(top=0.93)
        t= f.suptitle('Attributes Correlation Matrix', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)
        fu.log('Saving Correlation Matrix Plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)


        return 0

def main():
    parser = argparse.ArgumentParser(description="Generates a correlation matrix from a given dataset", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_plot_path', required=True, help='Path to the correlation matrix plot. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    CorrelationMatrix(input_dataset_path=args.input_dataset_path,
                   output_plot_path=args.output_plot_path,
                   properties=properties).launch()

if __name__ == '__main__':
    main()

