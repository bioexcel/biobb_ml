#!/usr/bin/env python3

"""Module containing the AgglClustering class and the command line interface."""
import argparse
import io
from sklearn.preprocessing import StandardScaler
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.clustering.common import *


class AgglClustering():
    """Clusters a given dataset with agglomerative clustering method.
    Wrapper of the sklearn.cluster.AgglomerativeClustering module
    Visit the `sklearn official website <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_agglomerative_clustering.csv>`_. Accepted formats: csv.
        output_results_path (str): Path to the clustered dataset. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_agglomerative_clustering.csv>`_. Accepted formats: csv.
        output_plot_path (str) (Optional): Path to the clustering plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_agglomerative_clustering.png>`_. Accepted formats: png.
        properties (dic):
            * **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
            * **clusters** (*int*) - (3) The number of clusters to form as well as the number of centroids to generate.
            * **linkage** (*int*) - ("ward") The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. Values: ward, complete, average, single.
            * **plots** (*list*) - (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ].
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
        self.clusters = properties.get('clusters', 3)
        self.linkage = properties.get('linkage', 'ward')
        self.plots = properties.get('plots', [])
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
        """Launches the execution of the AgglClustering module."""

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

        # create an agglomerative clustering object with self.clusters clusters
        model = AgglomerativeClustering(n_clusters = self.clusters, linkage = self.linkage)
        # fit the data
        model.fit(t_predictors)

        # create a copy of data, so we can see the clusters next to the original data
        clusters = data.copy()
        # predict the cluster for each observation
        clusters['cluster'] = model.fit_predict(t_predictors)

        fu.log('Calculating results\n\nCLUSTERING TABLE\n\n%s\n' % clusters, out_log, self.global_log)

        # save results
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], out_log, self.global_log)
        clusters.to_csv(self.io_dict["out"]["output_results_path"], index = False, header=True, float_format='%.3f')

        if self.io_dict["out"]["output_plot_path"] and self.plots: 
            new_plots = []
            i = 0
            for plot in self.plots:
                if len(plot['features']) == 2 or len(plot['features']) == 3:
                    new_plots.append(plot)
                    i += 1
                if i == 6:
                    break

            plot = plotCluster(new_plots, clusters)
            fu.log('Saving output plot to %s' % self.io_dict["out"]["output_plot_path"], out_log, self.global_log)
            plot.savefig(self.io_dict["out"]["output_plot_path"], dpi=150)

        return 0

def main():
    parser = argparse.ArgumentParser(description="Clusters a given dataset with k-means clustering method.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the clustered dataset. Accepted formats: csv.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the clustering plot. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    AgglClustering(input_dataset_path=args.input_dataset_path,
                   output_results_path=args.output_results_path, 
                   output_plot_path=args.output_plot_path, 
                   properties=properties).launch()

if __name__ == '__main__':
    main()
