#!/usr/bin/env python3

"""Module containing the PairwiseComparison class and the command line interface."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from biobb_common.generic.biobb_object import BiobbObject
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_ml.utils.common import *


class PairwiseComparison(BiobbObject):
    """
    | biobb_ml PairwiseComparison
    | Generates a pairwise comparison from a given dataset.

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_pairwise_comparison.csv>`_. Accepted formats: csv (edam:format_3752).
        output_plot_path (str): Path to the pairwise comparison plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_plot_pairwise_comparison.png>`_. Accepted formats: png (edam:format_3603).
        properties (dic):
            * **features** (*dict*) - ({}) Independent variables or columns from your dataset you want to compare. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.utils.pairwise_comparison import pairwise_comparison
            prop = { 
                'features': {
                    'columns': [ 'column1', 'column2', 'column3' ] 
                }
            }
            pairwise_comparison(input_dataset_path='/path/to/myDataset.csv', 
                            output_plot_path='/path/to/newPlot.png', 
                            properties=prop)

    Info:
        * wrapped_software:
            * name: In house
            * license: Apache-2.0
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_dataset_path, output_plot_path, 
                properties=None, **kwargs) -> None:
        properties = properties or {}

        # Call parent class constructor
        super().__init__(properties)
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.features = properties.get('features', {})
        self.properties = properties

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

    def check_data_params(self, out_log, err_log):
        """ Checks all the input/output paths and parameters """
        self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", out_log, self.__class__.__name__)
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`PairwiseComparison <utils.pairwise_comparison.PairwiseComparison>` utils.pairwise_comparison.PairwiseComparison object."""

        # check input/output paths and parameters
        self.check_data_params(self.out_log, self.err_log)

        # Setup Biobb
        if self.check_restart(): return 0
        self.stage_files()

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], self.out_log, self.global_log)
        if 'columns' in self.features:
            labels = getHeader(self.io_dict["in"]["input_dataset_path"])
            skiprows = 1
        else:
            labels = None
            skiprows = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = None, sep="\\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)

        fu.log('Parsing [%s] columns of the dataset' % getIndependentVarsList(self.features), self.out_log, self.global_log)
        if not self.features: cols = data[data.columns]
        else: cols = getIndependentVars(self.features, data, self.out_log, self.__class__.__name__)
        pp = sns.pairplot(cols, height=1.8, aspect=1.8,
                          plot_kws=dict(edgecolor="k", linewidth=0.5),
                          diag_kind="kde", diag_kws=dict(shade=True))
        fig = pp.fig 
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle('Attributes Pairwise Plots', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)
        fu.log('Saving Pairwise Plot to %s' % self.io_dict["out"]["output_plot_path"], self.out_log, self.global_log)

        # Copy files to host
        self.copy_to_host()

        self.tmp_files.extend([
            self.stage_io_dict.get("unique_dir")
        ])
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)

        return 0

def pairwise_comparison(input_dataset_path: str, output_plot_path: str, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`PairwiseComparison <utils.pairwise_comparison.PairwiseComparison>` class and
    execute the :meth:`launch() <utils.pairwise_comparison.PairwiseComparison.launch>` method."""

    return PairwiseComparison(input_dataset_path=input_dataset_path, 
                   output_plot_path=output_plot_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Generates a pairwise comparison from a given dataset", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_plot_path', required=True, help='Path to the pairwise comparison plot. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    pairwise_comparison(input_dataset_path=args.input_dataset_path,
                       output_plot_path=args.output_plot_path,
                       properties=properties)

if __name__ == '__main__':
    main()

