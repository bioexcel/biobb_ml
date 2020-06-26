#!/usr/bin/env python3

"""Module containing the KMeansCoefficient class and the command line interface."""
import argparse
import io
from sklearn.preprocessing import StandardScaler
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.clustering.common import *


class KMeansCoefficient():
    """Clusters a given dataset and calculates best K coefficient for a k-means clustering.
    Wrapper of the sklearn.cluster.KMeans module
    Visit the `sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_k_means_coefficient.csv>`_. Accepted formats: csv.
        output_results_path (str): Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_k_means_coefficient.csv>`_. Accepted formats: csv.
        output_plot_path (str) (Optional): Path to the elbow method and gap statistics plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_k_means_coefficient.png>`_. Accepted formats: png.
        properties (dic):
            * **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
            * **max_clusters** (*int*) - (6) Maximum number of clusters to use by default for kmeans queries.
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
        self.predictors = properties.get('predictors', [])
        self.max_clusters = properties.get('max_clusters', 6)
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
        """Launches the execution of the KMeansCoefficient module."""

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

        # the features are the predictors
        predictors = data.filter(self.predictors)

        # Hopkins test
        H = hopkins(predictors)
        fu.log('Performing Hopkins test over dataset. H = %f' % H, out_log, self.global_log)

        # scale dataset
        fu.log('Scaling dataset', out_log, self.global_log)
        scaler = StandardScaler()
        t_predictors = scaler.fit_transform(predictors)

        # calculate wcss for each cluster
        fu.log('Calculating Within-Clusters Sum of Squares (WCSS) for each %d clusters' % self.max_clusters, out_log, self.global_log)
        wcss = getWCSS('kmeans', self.max_clusters, t_predictors)
            
        # wcss table
        wcss_table = pd.DataFrame(data={'cluster': np.arange(1, self.max_clusters + 1), 'WCSS': wcss})
        fu.log('Calculating WCSS for each cluster\n\nWCSS TABLE\n\n%s\n' % wcss_table.to_string(index=False), out_log, self.global_log)

        # get best cluster elbow method
        best_k, elbow_index = get_best_K(wcss)
        fu.log('Optimal number of clusters according to the Elbow Method is %d' % best_k, out_log, self.global_log)

        # calculate gap
        best_g, gap = getGap('kmeans', t_predictors, nrefs=5, maxClusters=(self.max_clusters + 1))

        # gap table
        gap_table = pd.DataFrame(data={'cluster': np.arange(1, self.max_clusters + 1), 'GAP': gap['gap']})
        fu.log('Calculating Gap for each cluster\n\nGAP TABLE\n\n%s\n' % gap_table.to_string(index=False), out_log, self.global_log)

        # log best cluster gap method
        fu.log('Optimal number of clusters according to the Gap Statistics Method is %d' % best_g, out_log, self.global_log)

        # calculate silhouette
        silhouette_list, s_list = getSilhouetthe('kmeans', t_predictors, self.max_clusters)

        # silhouette table
        silhouette_table = pd.DataFrame(data={'cluster': np.arange(1, self.max_clusters + 1), 'SILHOUETTE': silhouette_list})
        fu.log('Calculating Silhouette for each cluster\n\nSILHOUETTE TABLE\n\n%s\n' % silhouette_table.to_string(index=False), out_log, self.global_log)

        # get best cluster silhouette method
        key = silhouette_list.index(max(silhouette_list))
        best_s = s_list.__getitem__(key)
        fu.log('Optimal number of clusters according to the Silhouette Method is %d' % best_s, out_log, self.global_log)

        # save results table
        results_table = pd.DataFrame(data={'method': ['elbow', 'gap', 'silhouette'], 'coefficient': [wcss[elbow_index], max(gap['gap']) ,max(silhouette_list)], 'clusters': [best_k, best_g ,best_s]})
        fu.log('Gathering results\n\nRESULTS TABLE\n\n%s\n' % results_table.to_string(index=False), out_log, self.global_log)
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        results_table.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        # wcss plot
        if self.io_dict["out"]["output_plot_path"]: 
            fu.log('Saving methods plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot = plotKmeansTrain(self.max_clusters, wcss, gap['gap'], silhouette_list, best_k, best_g, best_s)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Clusters a given dataset and calculates best K coefficient for a k-means clustering.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
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
    KMeansCoefficient(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()
