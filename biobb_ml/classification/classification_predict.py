#!/usr/bin/env python3

"""Module containing the ClassificationPredict class and the command line interface."""
import argparse
import pandas as pd
import joblib
from biobb_common.generic.biobb_object import BiobbObject
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import svm
from biobb_common.configuration import settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_ml.classification.common import check_input_path, check_output_path, getHeader, get_list_of_predictors, get_keys_of_predictors


class ClassificationPredict(BiobbObject):
    """
    | biobb_ml ClassificationPredict
    | Makes predictions from an input dataset and a given classification model.
    | Makes predictions from an input dataset (provided either as a file or as a dictionary property) and a given classification model trained with `DecisionTreeClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_, `KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_, `LogisticRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_, `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_, `Support Vector Machine <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ methods.

    Args:
        input_model_path (str): Path to the input model. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/model_classification_predict.pkl>`_. Accepted formats: pkl (edam:format_3653).
        input_dataset_path (str) (Optional): Path to the dataset to predict. File type: input. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/input_classification_predict.csv>`_. Accepted formats: csv (edam:format_3752).
        output_results_path (str): Path to the output results file. File type: output. `Sample file <https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_classification_predict.csv>`_. Accepted formats: csv (edam:format_3752).
        properties (dic - Python dictionary object containing the tool parameters, not input/output files):
            * **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets. It will be taken into account only in case **input_dataset_path** is not provided. Format: [{ 'var1': 1.0, 'var2': 2.0 }, { 'var1': 4.0, 'var2': 2.7 }] for datasets with headers and [[ 1.0, 2.0 ], [ 4.0, 2.7 ]] for datasets without headers.
            * **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.
            * **sandbox_path** (*str*) - ("./") [WF property] Parent path to the sandbox directory.

    Examples:
        This is a use example of how to use the building block from Python::

            from biobb_ml.classification.classification_predict import classification_predict
            prop = {
                'predictions': [
                    {
                        'var1': 1.0,
                        'var2': 2.0
                    },
                    {
                        'var1': 4.0,
                        'var2': 2.7
                    }
                ]
            }
            classification_predict(input_model_path='/path/to/myModel.pkl',
                                    output_results_path='/path/to/newPredictedResults.csv',
                                    input_dataset_path='/path/to/myDataset.csv',
                                    properties=prop)

    Info:
        * wrapped_software:
            * name: scikit-learn
            * version: >=0.24.2
            * license: BSD 3-Clause
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl

    """

    def __init__(self, input_model_path, output_results_path,
                 input_dataset_path=None, properties=None, **kwargs) -> None:
        properties = properties or {}

        # Call parent class constructor
        super().__init__(properties)
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = {
            "in": {"input_model_path": input_model_path, "input_dataset_path": input_dataset_path},
            "out": {"output_results_path": output_results_path}
        }

        # Properties specific for BB
        self.predictions = properties.get('predictions', [])
        self.properties = properties

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

    def check_data_params(self, out_log, err_log):
        """ Checks all the input/output paths and parameters """
        self.io_dict["in"]["input_model_path"] = check_input_path(self.io_dict["in"]["input_model_path"], "input_model_path", out_log, self.__class__.__name__)
        self.io_dict["out"]["output_results_path"] = check_output_path(self.io_dict["out"]["output_results_path"], "output_results_path", False, out_log, self.__class__.__name__)
        if self.io_dict["in"]["input_dataset_path"]:
            self.io_dict["in"]["input_dataset_path"] = check_input_path(self.io_dict["in"]["input_dataset_path"], "input_dataset_path", out_log, self.__class__.__name__)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`ClassificationPredict <classification.classification_predict.ClassificationPredict>` classification.classification_predict.ClassificationPredict object."""

        # check input/output paths and parameters
        self.check_data_params(self.out_log, self.err_log)

        # Setup Biobb
        if self.check_restart():
            return 0
        self.stage_files()

        fu.log('Getting model from %s' % self.io_dict["in"]["input_model_path"], self.out_log, self.global_log)

        with open(self.io_dict["in"]["input_model_path"], "rb") as f:
            while True:
                try:
                    m = joblib.load(f)
                    if (isinstance(m, linear_model.LogisticRegression) or isinstance(m, KNeighborsClassifier) or isinstance(m, DecisionTreeClassifier) or isinstance(m, ensemble.RandomForestClassifier) or isinstance(m, svm.SVC)):
                        new_model = m
                    if isinstance(m, StandardScaler):
                        scaler = m
                    if isinstance(m, dict):
                        variables = m
                except EOFError:
                    break

        if self.io_dict["in"]["input_dataset_path"]:
            # load dataset from input_dataset_path file
            fu.log('Getting dataset from %s' % self.io_dict["in"]["input_dataset_path"], self.out_log, self.global_log)
            if 'columns' in variables['independent_vars']:
                labels = getHeader(self.io_dict["in"]["input_dataset_path"])
                skiprows = 1
            else:
                labels = None
                skiprows = None
            new_data_table = pd.read_csv(self.io_dict["in"]["input_dataset_path"], header=None, sep="\\s+|;|:|,|\t", engine="python", skiprows=skiprows, names=labels)
        else:
            # load dataset from properties
            if 'columns' in variables['independent_vars']:
                # sorting self.properties in the correct order given by variables['independent_vars']['columns']
                index_map = {v: i for i, v in enumerate(variables['independent_vars']['columns'])}
                predictions = []
                for i, pred in enumerate(self.predictions):
                    sorted_pred = sorted(pred.items(), key=lambda pair: index_map[pair[0]])
                    predictions.append(dict(sorted_pred))
                new_data_table = pd.DataFrame(data=get_list_of_predictors(predictions), columns=get_keys_of_predictors(predictions))
            else:
                predictions = self.predictions
                new_data_table = pd.DataFrame(data=predictions)

        if variables['scale']:
            fu.log('Scaling dataset', self.out_log, self.global_log)
            new_data = scaler.transform(new_data_table)
        else:
            new_data = new_data_table

        p = new_model.predict_proba(new_data)

        # if headers, create target column with proper label
        if self.io_dict["in"]["input_dataset_path"] or 'columns' in variables['independent_vars']:
            clss = ' (' + ', '.join(str(x) for x in variables['target_values']) + ')'
            new_data_table[variables['target']['column'] + ' ' + clss] = tuple(map(tuple, p))
        else:
            new_data_table[len(new_data_table.columns)] = tuple(map(tuple, p))
        fu.log('Predicting results\n\nPREDICTION RESULTS\n\n%s\n' % new_data_table, self.out_log, self.global_log)
        fu.log('Saving results to %s' % self.io_dict["out"]["output_results_path"], self.out_log, self.global_log)
        new_data_table.to_csv(self.io_dict["out"]["output_results_path"], index=False, header=True, float_format='%.3f')

        # Copy files to host
        self.copy_to_host()

        self.tmp_files.extend([
            self.stage_io_dict.get("unique_dir")
        ])
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)

        return 0


def classification_predict(input_model_path: str, output_results_path: str, input_dataset_path: str = None, properties: dict = None, **kwargs) -> int:
    """Execute the :class:`ClassificationPredict <classification.classification_predict.ClassificationPredict>` class and
    execute the :meth:`launch() <classification.classification_predict.ClassificationPredict.launch>` method."""

    return ClassificationPredict(input_model_path=input_model_path,
                                 output_results_path=output_results_path,
                                 input_dataset_path=input_dataset_path,
                                 properties=properties, **kwargs).launch()


def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description="Makes predictions from an input dataset and a given classification model.", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('--config', required=False, help='Configuration file')

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_model_path', required=True, help='Path to the input model. Accepted formats: pkl.')
    required_args.add_argument('--output_results_path', required=True, help='Path to the output results file. Accepted formats: csv.')
    parser.add_argument('--input_dataset_path', required=False, help='Path to the dataset to predict. Accepted formats: csv.')

    args = parser.parse_args()
    args.config = args.config or "{}"
    properties = settings.ConfReader(config=args.config).get_prop_dic()

    # Specific call of each building block
    classification_predict(input_model_path=args.input_model_path,
                           output_results_path=args.output_results_path,
                           input_dataset_path=args.input_dataset_path,
                           properties=properties)


if __name__ == '__main__':
    main()
