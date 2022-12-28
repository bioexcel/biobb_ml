#!/usr/bin/env python3

"""Module containing the DropColumns class and the command line interface."""
import argparse
import pandas as pd
from biobb_common.generic.biobb_object import BiobbObject
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_ml.utils.common import *


class DropColumns(BiobbObject):
    """
    | biobb_ml DropColumns
    | Drops columns from a given dataset.

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_drop.csv>`_. Accepted formats: csv (edam:format_3752).
        output_dataset_path (str): Path to the output dataset. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_drop.csv>`_. Accepted formats: csv (edam:format_3752).
        properties (dic):
            * **targets** (*dict*) - ({}) Independent variables or columns from your dataset you want to drop. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.utils.drop_columns import drop_columns
            prop = { 
                'targets': {
                    'columns': [ 'column1', 'column2', 'column3' ] 
                }
            }
            drop_columns(input_dataset_path='/path/to/myDataset.csv', 
                            output_dataset_path='/path/to/newDataset.csv', 
                            properties=prop)

    Info:
        * wrapped_software:
            * name: In house
            * license: Apache-2.0
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_dataset_path,  output_dataset_path, 
                properties=None, **kwargs) -> None:
        properties = properties or {}

        # Call parent class constructor
        super().__init__(properties)
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_dataset_path": output_dataset_path } 
        }

        # Properties specific for BB
        self.targets = properties.get('targets', {})
        self.properties = properties

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

    def check_data_params(self, out_log, err_log):
        """ Checks all the input/output paths and parameters """
        self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", out_log, self.__class__.__name__)
        self.io_dict["out"]["output_dataset_path"] = check_output_path(self.io_dict["out"]["output_dataset_path"],"output_dataset_path", False, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`DropColumns <utils.drop_columns.DropColumns>` utils.drop_columns.DropColumns object."""

        # check input/output paths and parameters
        self.check_data_params(self.out_log, self.err_log)

        # Setup Biobb
        if self.check_restart(): return 0
        self.stage_files()

        # load dataset
        fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], self.out_log, self.global_log)
        if 'columns' in self.targets:
            labels = getHeader(self.io_dict["in"]["input_dataset_path"])
            skiprows = 1
            header = 0
        else:
            labels = None
            skiprows = None
            header = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = None, sep="\\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)

        targets = getTargetsList(self.targets, 'drop', self.out_log, self.__class__.__name__)

        fu.log('Dropping [%s] columns from dataset' % getIndependentVarsList(self.targets), self.out_log, self.global_log)
        data = data.drop(targets, axis=1)

        hdr = False
        if header == 0: hdr = True
        fu.log('Saving dataset to %s' % self.io_dict["out"]["output_dataset_path"], self.out_log, self.global_log)
        data.to_csv(self.io_dict["out"]["output_dataset_path"], index = False, header=hdr)

        # Copy files to host
        self.copy_to_host()

        self.tmp_files.extend([
            self.stage_io_dict.get("unique_dir")
        ])
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)

        return 0

def drop_columns(input_dataset_path: str, output_dataset_path: str, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`DropColumns <utils.drop_columns.DropColumns>` class and
    execute the :meth:`launch() <utils.drop_columns.DropColumns.launch>` method."""

    return DropColumns(input_dataset_path=input_dataset_path, 
                   output_dataset_path=output_dataset_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Drops columns from a given dataset.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_dataset_path', required=True, help='Path to the output dataset. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    drop_columns(input_dataset_path=args.input_dataset_path,
                   output_dataset_path=args.output_dataset_path,
                   properties=properties)

if __name__ == '__main__':
    main()

