#!/usr/bin/env python3

"""Module containing the KMeansClustering class and the command line interface."""
import argparse
import io
import joblib
from sklearn.preprocessing import StandardScaler
from biobb_common.configuration import  settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.command_wrapper import cmd_wrapper
from biobb_ml.clustering.common import *


class KMeansClustering():
    """
    | biobb_ml KMeansClustering
    | Wrapper of the scikit-learn KMeans method. 
    | Clusters a given dataset and saves the model and scaler. Visit the `KMeans documentation page <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ in the sklearn official website for further information. 

    Args:
        input_dataset_path (str): Path to the input dataset. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_k_means.csv>`_. Accepted formats: csv (edam:format_3752).
        output_results_path (str): Path to the clustered dataset. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_k_means.csv>`_. Accepted formats: csv (edam:format_3752).
        output_model_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_model_k_means.pkl>`_. Accepted formats: pkl (edam:format_3653).
        output_plot_path (str) (Optional): Path to the clustering plot. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_k_means.png>`_. Accepted formats: png (edam:format_3603).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **predictors** (*dict*) - ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked.
            * **clusters** (*int*) - (3) [1~100|1] The number of clusters to form as well as the number of centroids to generate.
            * **plots** (*list*) - (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ].
            * **random_state_method** (*int*) - (5) [1~1000|1] Determines random number generation for centroid initialization.
            * **scale** (*bool*) - (False) Whether or not to scale the input dataset.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.clustering.k_means import k_means
            prop = { 
                'predictors': { 
                    'columns': [ 'column1', 'column2', 'column3' ] 
                }, 
                'clusters': 3, 
                'plots': [ 
                    { 
                        'title': 'Plot 1', 
                        'features': ['feat1', 'feat2'] 
                    } 
                ] 
            }
            k_means(input_dataset_path='/path/to/myDataset.csv', 
                    output_results_path='/path/to/newTable.csv', 
                    output_model_path='/path/to/newModel.pkl', 
                    output_plot_path='/path/to/newPlot.png', 
                    properties=prop)

    Info:
        * wrapped_software:
            * name: scikit-learn KMeans
            * version: >=0.23.1
            * license: BSD 3-Clause
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_dataset_path, output_results_path, output_model_path, 
                output_plot_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Input/Output files
        self.io_dict = { 
            "in": { "input_dataset_path": input_dataset_path }, 
            "out": { "output_results_path": output_results_path, "output_model_path": output_model_path, "output_plot_path": output_plot_path } 
        }

        # Properties specific for BB
        self.predictors = properties.get('predictors', {})
        self.clusters = properties.get('clusters', 3)
        self.plots = properties.get('plots', [])
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
        self.io_dict["out"]["output_model_path"] = check_output_path(self.io_dict["out"]["output_model_path"],"output_model_path", False, out_log, self.__class__.__name__)
        if self.io_dict["out"]["output_plot_path"]:
            self.io_dict["out"]["output_plot_path"] = check_output_path(self.io_dict["out"]["output_plot_path"],"output_plot_path", True, out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`KMeansClustering <clustering.k_means.KMeansClustering>` clustering.k_means.KMeansClustering object."""

        # Get local loggers from launchlogger decorator
        out_log = getattr(self, 'out_log', None)
        err_log = getattr(self, 'err_log', None)

        # check input/output paths and parameters
        self.check_data_params(out_log, err_log)

        # Check the properties
        fu.check_properties(self, self.properties)

        if self.restart:
            output_file_list = [self.io_dict["out"]["output_results_path"],self.io_dict["out"]["output_model_path"],self.io_dict["out"]["output_plot_path"]]
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

        # create a k-means object with self.clusters clusters
        model = KMeans(n_clusters=self.clusters, random_state=self.random_state_method)
        # fit the data
        model.fit(predictors)

        # create a copy of data, so we can see the clusters next to the original data
        clusters = data.copy()
        # predict the cluster for each observation
        clusters['cluster'] = model.predict(predictors)

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

        # save model, scaler and parameters
        variables = {
            'predictors': self.predictors,
            'scale': self.scale,
        }
        fu.log('Saving model to %s' % self.io_dict["out"]["output_model_path"], out_log, self.global_log)
        with open(self.io_dict["out"]["output_model_path"], "wb") as f:
            joblib.dump(model, f)
            if self.scale: joblib.dump(scaler, f)
            joblib.dump(variables, f)

        return 0

def k_means(input_dataset_path: str, output_results_path: str, output_model_path: str, output_plot_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`KMeansClustering <clustering.k_means.KMeansClustering>` class and
    execute the :meth:`launch() <clustering.k_means.KMeansClustering.launch>` method."""

    return KMeansClustering(input_dataset_path=input_dataset_path,  
                   output_results_path=output_results_path, 
                   output_model_path=output_model_path, 
                   output_plot_path=output_plot_path,
                   properties=properties, **kwargs).launch()

def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Wrapper of the scikit-learn KMeans method.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_dataset_path', required=True, help='Path to the input dataset. Accepted formats: csv.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the clustered dataset. Accepted formats: csv.')
    required_args.add_argument('--output_model_path', required=True, help='Path to the output model file. Accepted formats: pkl.')
    parser.add_argument('--output_plot_path', required=False, help='Path to the clustering plot. Accepted formats: png.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    k_means(input_dataset_path=args.input_dataset_path,
           output_results_path=args.output_results_path, 
           output_model_path=args.output_model_path, 
           output_plot_path=args.output_plot_path, 
           properties=properties)

if __name__ == '__main__':
    main()