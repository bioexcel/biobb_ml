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
    """Analyses a given dataset through Principal Component Analysis (PCA).
    Wrapper of the sklearn.decomposition.PCA module
    Visit the `sklearn official website <https://scikit-learn.org/0.16/modules/generated/sklearn.decomposition.PCA.html>`_. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/dimensionality_reduction/dataset_principal_component.csv>`_. Accepted formats: csv.
        output_results_path (str): Path to the analysed dataset. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_results_principal_component.csv>`_. Accepted formats: csv.
        output_plot_path (str) (Optional): Path to the Principal Component plot, only if number of components is 2 or 3. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_plot_principal_component.png>`_. Accepted formats: png.
        properties (dic):
            * **features** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
            * **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
            * **n_components** (*int* / *float*) - (None) Number of components to keep (int) or minimum number of principal components such the 0 to 1 range of the variance (float) is retained. If n_components is not set (None) all components are kept.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
    """

    def __init__(self, input_dataset_path,
                 output_results_path, output_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.features = properties.get('features', [])
        self.target = properties.get('target', '')
        self.n_components = properties.get('n_components', None)
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
        self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Launches the execution of the PrincipalComponentAnalysis module."""

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
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"])

        # get features
        features = data.filter(self.features)

        # scale dataset
        fu.log('Scaling dataset', out_log, self.global_log)
        scaler = StandardScaler()
        t_features = scaler.fit_transform(features)

        # create a PCA object with self.n_components n_components
        fu.log('Fitting dataset', out_log, self.global_log)
        model = PCA(n_components = self.n_components)
        # fit the data
        model.fit(t_features)

        # calculate variance ratio
        v_ratio = model.explained_variance_ratio_
        fu.log('Variance ratio for %d Principal Components: %s' % (v_ratio.shape[0], np.array2string(v_ratio, precision=3, separator=', ')), out_log, self.global_log)

        # transform
        fu.log('Transforming dataset', out_log, self.global_log)
        pca = model.transform(t_features)
        pca = pd.DataFrame(data = pca, columns = generate_columns_labels('PC', v_ratio.shape[0]))

        # output results
        pca_table = pd.concat([pca, data[[self.target]]], axis = 1)
        fu.log('Calculating PCA for dataset\n\n%d COMPONENT PCA TABLE\n\n%s\n' % (v_ratio.shape[0], pca_table), out_log, self.global_log)

        # save results
        fu.log('Saving data to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        pca_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True)

        # create output plot
        if(self.io_dict["out"]["output_plot_path"]): 
            if v_ratio.shape[0] > 3: 
                fu.log('%d PC\'s found. Displaying only 1st, 2nd and 3rd PC' % v_ratio.shape[0], out_log, self.global_log)
            fu.log('Saving PC plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            targets = np.unique(data[[self.target]])
            if v_ratio.shape[0] == 2:
                PCA2CPlot(pca_table, targets, self.target)

            if v_ratio.shape[0] >= 3:
                PCA3CPlot(pca_table, targets, self.target)

            plt.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Analyses a given dataset through Principal Component Analysis (PCA).", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
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
    PrincipalComponentAnalysis(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()
