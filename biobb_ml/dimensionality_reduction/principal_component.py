#!/usr/bin/env python3

"""Module containing the PrincipalComponentAnalysis class and the command line interface."""
import argparse
import io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.dimensionality_reduction.common import *


class PrincipalComponentAnalysis():
    """
    | biobb_ml PrincipalComponentAnalysis
    | Wrapper of the scikit-learn PCA method. 
    | Analyses a given dataset through Principal Component Analysis (PCA). Visit the `PCA documentation page <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ in the sklearn official website for further information. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/dimensionality_reduction/dataset_principal_component.csv>`_. Accepted formats: csv (edam:format_3752).
        output_results_path (str): Path to the analysed dataset. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_results_principal_component.csv>`_. Accepted formats: csv (edam:format_3752).
        output_plot_path (str) (Optional): Path to the Principal Component plot, only if number of components is 2 or 3. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_plot_principal_component.png>`_. Accepted formats: png (edam:format_3603).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **features** (*dict*) - ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **target** (*dict*) - ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked.
            * **n_components** (*dict*) - ({}) Dictionary containing the number of components to keep (int) or the minimum number of principal components such the 0 to 1 range of the variance (float) is retained. If not set ({}) all components are kept. Formats for integer values: { "value": 2 } or for float values: { "value": 0.3 }
            * **random_state_method** (*int*) - (5) [1~1000|1] Controls the randomness of the estimator.
            * **scale** (*bool*) - (False) Whether or not to scale the input dataset.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    
    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.dimensionality_reduction.principal_component import principal_component
            prop = { 
                'features': { 
                    'columns': [ 'column1', 'column2', 'column3' ] 
                }, 
                'target': { 
                    'column': 'target' 
                }, 
                'n_components': { 
                    'int': 2 
                } 
            }
            principal_component(input_dataset_path='/path/to/myDataset.csv', 
                                output_results_path='/path/to/newTable.csv', 
                                output_plot_path='/path/to/newPlot.png', 
                                properties=prop)

    Info:
        * wrapped_software:
            * name: scikit-learn PCA
            * version: >=0.23.1
            * license: BSD 3-Clause
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_dataset_path, output_results_path, 
                output_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.features = properties.get('features', {})
        self.target = properties.get('target', {})
        self.n_components = properties.get('n_components', {})
        self.random_state_method = properties.get('random_state_method', 5)
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
        self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", out_log, self.__class__.__name__)
        self.io_dict["out"]["output_results_path"] = check_output_path(self.io_dict["out"]["output_results_path"],"output_results_path", False, out_log, self.__class__.__name__)
        if self.io_dict["out"]["output_plot_path"]:
            self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`PrincipalComponentAnalysis <dimensionality_reduction.principal_component.PrincipalComponentAnalysis>` dimensionality_reduction.pincipal_component.PrincipalComponentAnalysis object."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_results_path"],self.io_dict["out"]["output_plot_path"]]
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

        # declare inputs, targets and weights
        # the inputs are all the features
        features = getIndependentVars(self.features, data, out_log, self.__class__.__name__)
        fu.log('Features: [%s]' % (getIndependentVarsList(self.features)), out_log, self.global_log)
        # target
        y_value = getTargetValue(self.target)
        fu.log('Target: %s' % (y_value), out_log, self.global_log)

        if self.scale: 
            fu.log('Scaling dataset', out_log, self.global_log)
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

        # create a PCA object with self.n_components['value'] n_components
        if not 'value' in self.n_components:
            n_c = None
        else:
            n_c = self.n_components['value']
        fu.log('Fitting dataset', out_log, self.global_log)
        model = PCA(n_components = n_c, random_state = self.random_state_method)
        # fit the data
        model.fit(features)

        # calculate variance ratio
        v_ratio = model.explained_variance_ratio_
        fu.log('Variance ratio for %d Principal Components: %s' % (v_ratio.shape[0], np.array2string(v_ratio, precision=3, separator=', ')), out_log, self.global_log)

        # transform
        fu.log('Transforming dataset', out_log, self.global_log)
        pca = model.transform(features)
        pca = pd.DataFrame(data = pca, columns = generate_columns_labels('PC', v_ratio.shape[0]))

        if 'columns' in self.features:
            d = data[[y_value]]
            target_plot = y_value
        else:
            d = data.loc[:,int(y_value)]
            target_plot = int(y_value)

        # output results
        pca_table = pd.concat([pca, d], axis = 1)
        fu.log('Calculating PCA for dataset\n\n%d COMPONENT PCA TABLE\n\n%s\n' % (v_ratio.shape[0], pca_table), out_log, self.global_log)

        # save results
        fu.log('Saving data to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        pca_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True)

        # create output plot
        if(self.io_dict["out"]["output_plot_path"]): 
            if v_ratio.shape[0] > 3: 
                fu.log('%d PC\'s found. Displaying only 1st, 2nd and 3rd PC' % v_ratio.shape[0], out_log, self.global_log)
            fu.log('Saving PC plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            targets = np.unique(d)
            if v_ratio.shape[0] == 2:
                PCA2CPlot(pca_table, targets, target_plot)

            if v_ratio.shape[0] >= 3:
                PCA3CPlot(pca_table, targets, target_plot)

            plt.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        return 0

def principal_component(input_dataset_path: str, output_results_path: str, output_plot_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`PrincipalComponentAnalysis <dimensionality_reduction.principal_component.PrincipalComponentAnalysis>` class and
    execute the :meth:`launch() <dimensionality_reduction.principal_component.PrincipalComponentAnalysis.launch>` method."""

    return PrincipalComponentAnalysis(input_dataset_path=input_dataset_path,  
                   output_results_path=output_results_path, 
                   output_plot_path=output_plot_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of the scikit-learn PCA method.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the analysed dataset. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the Principal Component plot, only if number of components is 2 or 3. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    principal_component(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties)

if __name__ == '__main__':
    main()
