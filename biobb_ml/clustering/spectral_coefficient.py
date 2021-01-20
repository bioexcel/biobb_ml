#!/usr/bin/env python3

"""Module containing the SpectralCoefficient class and the command line interface."""
import argparse
import io
from sklearn.preprocessing import StandardScaler
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.clustering.common import *


class SpectralCoefficient():
    """
    | biobb_ml SpectralCoefficient
    | Wrapper of the scikit-learn SpectralClustering method. 
    | Clusters a given dataset and calculates best K coefficient. Visit the `SpectralClustering documentation page <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html>`_ in the sklearn official website for further information. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_spectral_coefficient.csv>`_. Accepted formats: csv (edam:format_3752).
        output_results_path (str): Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_spectral_coefficient.csv>`_. Accepted formats: csv (edam:format_3752).
        output_plot_path (str) (Optional): Path to the elbow method and gap statistics plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_spectral_coefficient.png>`_. Accepted formats: png (edam:format_3603).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **predictors** (*dict*) - ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **max_clusters** (*int*) - (6) [1~100|1] Maximum number of clusters to use by default for kmeans queries.
            * **random_state_method** (*int*) - (5) [1~1000|1] A pseudo random number generator used for the initialization of the lobpcg eigen vectors decomposition when *eigen_solver='amg'* and by the K-Means initialization.
            * **scale** (*bool*) - (False) Whether or not to scale the input dataset.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.clustering.spectral_coefficient import spectral_coefficient
            prop = { 
                'predictors': { 
                    'columns': [ 'column1', 'column2', 'column3' ] 
                }, 
                'max_clusters': 6 
            }
            spectral_coefficient(input_dataset_path='/path/to/myDataset.csv', 
                                output_results_path='/path/to/newTable.csv', 
                                output_plot_path='/path/to/newPlot.png', 
                                properties=prop)

    Info:
        * wrapped_software:
            * name: scikit-learn SpectralClustering
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
        self.predictors = properties.get('predictors', {})
        self.max_clusters = properties.get('max_clusters', 6)
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
        """Execute the :class:`SpectralCoefficient <clustering.spectral_coefficient.SpectralCoefficient>` clustering.spectral_coefficient.SpectralCoefficient object."""

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
        if 'columns' in self.predictors:
            labels = getHeader(self.io_dict["in"]["input_dataset_path"])
            skiprows = 1
        else:
            labels = None
            skiprows = None
        data = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header = None, sep="\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)

        # the features are the predictors
        predictors = getIndependentVars(self.predictors, data, out_log, self.__class__.__name__)
        fu.log('Predictors: [%s]' % (getIndependentVarsList(self.predictors)), out_log, self.global_log)

        # Hopkins test
        H = hopkins(predictors)
        fu.log('Performing Hopkins test over dataset. H = %f' % H, out_log, self.global_log)

        # scale dataset
        if self.scale: 
            fu.log('Scaling dataset', out_log, self.global_log)
            scaler = StandardScaler()
            predictors = scaler.fit_transform(predictors)

        # calculate silhouette
        silhouette_list, s_list = getSilhouetthe(method = 'spectral', X = predictors, max_clusters = self.max_clusters, random_state = self.random_state_method)

        # silhouette table
        silhouette_table = pd.DataFrame(data={'cluster': np.arange(1, self.max_clusters + 1), 'SILHOUETTE': silhouette_list})
        fu.log('Calculating Silhouette for each cluster\n\nSILHOUETTE TABLE\n\n%s\n' % silhouette_table.to_string(index=False), out_log, self.global_log)

        # get best cluster silhouette method
        key = silhouette_list.index(max(silhouette_list))
        best_s = s_list.__getitem__(key)
        fu.log('Optimal number of clusters according to the Silhouette Method is %d' % best_s, out_log, self.global_log)

        # save results table
        results_table = pd.DataFrame(data={'method': ['silhouette'], 'coefficient': [max(silhouette_list)], 'cluster': [best_s]})
        fu.log('Gathering results\n\nRESULTS TABLE\n\n%s\n' % results_table.to_string(index=False), out_log, self.global_log)
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        results_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        # wcss plot
        if self.io_dict["out"]["output_plot_path"]: 
            fu.log('Saving methods plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot = plotAgglomerativeTrain(self.max_clusters, silhouette_list, best_s)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        return 0

def spectral_coefficient(input_dataset_path: str, output_results_path: str, output_plot_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`SpectralCoefficient <clustering.spectral_coefficient.SpectralCoefficient>` class and
    execute the :meth:`launch() <clustering.spectral_coefficient.SpectralCoefficient.launch>` method."""

    return SpectralCoefficient(input_dataset_path=input_dataset_path,  
                   output_results_path=output_results_path, 
                   output_plot_path=output_plot_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of the scikit-learn SpectralClustering method.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the elbow and gap methods plot. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    spectral_coefficient(input_dataset_path=args.input_dataset_path,
                       output_results_path=args.output_results_path, 
                       output_plot_path=args.output_plot_path, 
                       properties=properties)

if __name__ == '__main__':
    main()
