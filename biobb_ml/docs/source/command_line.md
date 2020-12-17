# BioBB ML Command Line Help
Generic usage:
```python
biobb_command [-h] --config CONFIG --input_file(s) <input_file(s)> --output_file <output_file>
```
-----------------


## Principal_component
Wrapper of the scikit-learn PCA method.
### Get help
Command:
```python
principal_component -h
```
    usage: principal_component [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn PCA method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the Principal Component plot, only if number of components is 2 or 3. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the analysed dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/dimensionality_reduction/dataset_principal_component.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the analysed dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_results_principal_component.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the Principal Component plot, only if number of components is 2 or 3. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_plot_principal_component.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **n_components** (*object*): ({}) Dictionary containing the number of components to keep (int) or the minimum number of principal components such the 0 to 1 range of the variance (float) is retained. If not set ({}) all components are kept. Formats for integer values: { "value": 2 } or for float values: { "value": 0.3 }.
* **random_state_method** (*integer*): (5) Controls the randomness of the estimator..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_principal_component.yml)
```python
properties:
  features:
    columns:
    - sepal_length
    - sepal_width
    - petal_length
    - petal_width
  n_components:
    value: 2
  scale: true
  target:
    column: target

```
#### Command line
```python
principal_component --config config_principal_component.yml --input_dataset_path dataset_principal_component.csv --output_results_path ref_output_results_principal_component.csv --output_plot_path ref_output_plot_principal_component.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_principal_component.json)
```python
{
  "properties": {
    "features": {
      "columns": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
      ]
    },
    "target": {
      "column": "target"
    },
    "n_components": {
      "value": 2
    },
    "scale": true
  }
}
```
#### Command line
```python
principal_component --config config_principal_component.json --input_dataset_path dataset_principal_component.csv --output_results_path ref_output_results_principal_component.csv --output_plot_path ref_output_plot_principal_component.png
```

## Support_vector_machine
Wrapper of the scikit-learn SupportVectorMachine method.
### Get help
Command:
```python
support_vector_machine -h
```
    usage: support_vector_machine [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn SupportVectorMachine method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_support_vector_machine.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_support_vector_machine.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_support_vector_machine.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_support_vector_machine.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **kernel** (*string*): (rbf) Specifies the kernel type to be used in the algorithm. .
* **normalize_cm** (*boolean*): (False) Whether or not to normalize the confusion matrix..
* **random_state_method** (*integer*): (5) Controls the randomness of the estimator..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_support_vector_machine.yml)
```python
properties:
  independent_vars:
    range:
    - - 0
      - 2
    - - 4
      - 5
  kernel: rbf
  normalize_cm: false
  scale: true
  target:
    index: 6
  test_size: 0.2

```
#### Command line
```python
support_vector_machine --config config_support_vector_machine.yml --input_dataset_path dataset_support_vector_machine.csv --output_model_path ref_output_model_support_vector_machine.pkl --output_test_table_path ref_output_test_support_vector_machine.csv --output_plot_path ref_output_plot_support_vector_machine.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_support_vector_machine.json)
```python
{
  "properties": {
    "independent_vars": {
      "range": [
        [
          0,
          2
        ],
        [
          4,
          5
        ]
      ]
    },
    "target": {
      "index": 6
    },
    "kernel": "rbf",
    "normalize_cm": false,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
support_vector_machine --config config_support_vector_machine.json --input_dataset_path dataset_support_vector_machine.csv --output_model_path ref_output_model_support_vector_machine.pkl --output_test_table_path ref_output_test_support_vector_machine.csv --output_plot_path ref_output_plot_support_vector_machine.png
```

## Neural_network_predict
Makes predictions from an input dataset and a given classification model.
### Get help
Command:
```python
neural_network_predict -h
```
    usage: neural_network_predict [-h] [--config CONFIG] --input_model_path INPUT_MODEL_PATH --output_results_path OUTPUT_RESULTS_PATH [--input_dataset_path INPUT_DATASET_PATH]
    
    Makes predictions from an input dataset and a given classification model.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the dataset to predict. Accepted formats: csv.
    
    required arguments:
      --input_model_path INPUT_MODEL_PATH
                            Path to the input model. Accepted formats: h5.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the output results file. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_path** (*string*): Path to the input model. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/input_model_predict.h5). Accepted formats: H5
* **input_dataset_path** (*string*): Path to the dataset to predict. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_predict.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the output results file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_predict.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictions** (*array*): (None) List of dictionaries with all values you want to predict targets. It will be taken into account only in case **input_dataset_path** is not provided. Format: [{ 'var1': 1.0, 'var2': 2.0 }, { 'var1': 4.0, 'var2': 2.7 }] for datasets with headers and [[ 1.0, 2.0 ], [ 4.0, 2.7 ]] for datasets without headers..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_neural_network_predict.yml)
```python
properties:
  predictions:
  - AGE: 65.2
    LSTAT: 4.98
    RM: 6.575
    ZN: 18.0
  - AGE: 78.9
    LSTAT: 9.14
    RM: 6.421
    ZN: 0.0
  - AGE: 61.1
    LSTAT: 4.03
    RM: 7.185
    ZN: 0.0

```
#### Command line
```python
neural_network_predict --config config_neural_network_predict.yml --input_model_path input_model_predict.h5 --input_dataset_path dataset_predict.csv --output_results_path ref_output_predict.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_neural_network_predict.json)
```python
{
  "properties": {
    "predictions": [
      {
        "ZN": 18.0,
        "RM": 6.575,
        "AGE": 65.2,
        "LSTAT": 4.98
      },
      {
        "ZN": 0.0,
        "RM": 6.421,
        "AGE": 78.9,
        "LSTAT": 9.14
      },
      {
        "ZN": 0.0,
        "RM": 7.185,
        "AGE": 61.1,
        "LSTAT": 4.03
      }
    ]
  }
}
```
#### Command line
```python
neural_network_predict --config config_neural_network_predict.json --input_model_path input_model_predict.h5 --input_dataset_path dataset_predict.csv --output_results_path ref_output_predict.csv
```

## Regression_predict
Makes predictions from an input dataset and a given regression model.
### Get help
Command:
```python
regression_predict -h
```
    usage: regression_predict [-h] [--config CONFIG] --input_model_path INPUT_MODEL_PATH --output_results_path OUTPUT_RESULTS_PATH [--input_dataset_path INPUT_DATASET_PATH]
    
    Makes predictions from an input dataset and a given regression model.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the dataset to predict. Accepted formats: csv.
    
    required arguments:
      --input_model_path INPUT_MODEL_PATH
                            Path to the input model. Accepted formats: pkl.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the output results file. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_path** (*string*): Path to the input model. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/model_regression_predict.pkl). Accepted formats: PKL
* **input_dataset_path** (*string*): Path to the dataset to predict. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/input_classification_predict.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the output results file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_regression_predict.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictions** (*array*): (None) List of dictionaries with all values you want to predict targets. It will be taken into account only in case **input_dataset_path** is not provided. Format: [{ 'var1': 1.0, 'var2': 2.0 }, { 'var1': 4.0, 'var2': 2.7 }] for datasets with headers and [[ 1.0, 2.0 ], [ 4.0, 2.7 ]] for datasets without headers..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_regression_predict.yml)
```python
properties:
  predictions:
  - AGE: 65.2
    LSTAT: 4.98
    RM: 6.575
    ZN: 18.0
  - AGE: 78.9
    LSTAT: 9.14
    RM: 6.421
    ZN: 0.0

```
#### Command line
```python
regression_predict --config config_regression_predict.yml --input_model_path model_regression_predict.pkl --input_dataset_path input_classification_predict.csv --output_results_path ref_output_regression_predict.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_regression_predict.json)
```python
{
  "properties": {
    "predictions": [
      {
        "LSTAT": 4.98,
        "ZN": 18.0,
        "RM": 6.575,
        "AGE": 65.2
      },
      {
        "LSTAT": 9.14,
        "ZN": 0.0,
        "RM": 6.421,
        "AGE": 78.9
      }
    ]
  }
}
```
#### Command line
```python
regression_predict --config config_regression_predict.json --input_model_path model_regression_predict.pkl --input_dataset_path input_classification_predict.csv --output_results_path ref_output_regression_predict.csv
```

## K_means
Wrapper of the scikit-learn KMeans method.
### Get help
Command:
```python
k_means -h
```
    usage: k_means [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH --output_model_path OUTPUT_MODEL_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn KMeans method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the clustering plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the clustered dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_k_means.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_k_means.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_model_k_means.pkl). Accepted formats: PKL
* **output_plot_path** (*string*): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_k_means.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictors** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **clusters** (*integer*): (3) The number of clusters to form as well as the number of centroids to generate..
* **plots** (*array*): (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ]..
* **random_state_method** (*integer*): (5) Determines random number generation for centroid initialization..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_means.yml)
```python
properties:
  clusters: 3
  plots:
  - features:
    - sepal_length
    - sepal_width
    title: Plot 1
  - features:
    - petal_length
    - petal_width
    title: Plot 2
  - features:
    - sepal_length
    - sepal_width
    - petal_length
    title: Plot 3
  - features:
    - petal_length
    - petal_width
    - sepal_width
    title: Plot 4
  - features:
    - sepal_length
    - petal_width
    title: Plot 5
  predictors:
    columns:
    - sepal_length
    - sepal_width
    - petal_length
    - petal_width
  scale: true

```
#### Command line
```python
k_means --config config_k_means.yml --input_dataset_path dataset_k_means.csv --output_results_path ref_output_results_k_means.csv --output_model_path ref_output_model_k_means.pkl --output_plot_path ref_output_plot_k_means.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_means.json)
```python
{
  "properties": {
    "predictors": {
      "columns": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
      ]
    },
    "clusters": 3,
    "plots": [
      {
        "title": "Plot 1",
        "features": [
          "sepal_length",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 2",
        "features": [
          "petal_length",
          "petal_width"
        ]
      },
      {
        "title": "Plot 3",
        "features": [
          "sepal_length",
          "sepal_width",
          "petal_length"
        ]
      },
      {
        "title": "Plot 4",
        "features": [
          "petal_length",
          "petal_width",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 5",
        "features": [
          "sepal_length",
          "petal_width"
        ]
      }
    ],
    "scale": true
  }
}
```
#### Command line
```python
k_means --config config_k_means.json --input_dataset_path dataset_k_means.csv --output_results_path ref_output_results_k_means.csv --output_model_path ref_output_model_k_means.pkl --output_plot_path ref_output_plot_k_means.png
```

## K_neighbors
Wrapper of the scikit-learn KNeighborsClassifier method.
### Get help
Command:
```python
k_neighbors -h
```
    usage: k_neighbors [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn KNeighborsClassifier method. 
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_k_neighbors.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_k_neighbors.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_k_neighbors.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_k_neighbors.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **metric** (*string*): (minkowski) The distance metric to use for the tree. .
* **n_neighbors** (*integer*): (6) Number of neighbors to use by default for kneighbors queries..
* **normalize_cm** (*boolean*): (False) Whether or not to normalize the confusion matrix..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_neighbors.yml)
```python
properties:
  independent_vars:
    columns:
    - interest_rate
    - credit
    - march
    - previous
    - duration
  metric: minkowski
  n_neighbors: 5
  normalize_cm: false
  scale: true
  target:
    column: y
  test_size: 0.2

```
#### Command line
```python
k_neighbors --config config_k_neighbors.yml --input_dataset_path dataset_k_neighbors.csv --output_model_path ref_output_model_k_neighbors.pkl --output_test_table_path ref_output_test_k_neighbors.csv --output_plot_path ref_output_plot_k_neighbors.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_neighbors.json)
```python
{
  "properties": {
    "independent_vars": {
      "columns": [
        "interest_rate",
        "credit",
        "march",
        "previous",
        "duration"
      ]
    },
    "target": {
      "column": "y"
    },
    "metric": "minkowski",
    "n_neighbors": 5,
    "normalize_cm": false,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
k_neighbors --config config_k_neighbors.json --input_dataset_path dataset_k_neighbors.csv --output_model_path ref_output_model_k_neighbors.pkl --output_test_table_path ref_output_test_k_neighbors.csv --output_plot_path ref_output_plot_k_neighbors.png
```

## Logistic_regression
Wrapper of the scikit-learn LogisticRegression method.
### Get help
Command:
```python
logistic_regression -h
```
    usage: logistic_regression [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn LogisticRegression method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_logistic_regression.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_logistic_regression.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_logistic_regression.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_logistic_regression.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **solver** (*string*): (liblinear) Numerical optimizer to find parameters. .
* **c_parameter** (*number*): (0.01) Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization..
* **normalize_cm** (*boolean*): (False) Whether or not to normalize the confusion matrix..
* **random_state_method** (*integer*): (5) Controls the randomness of the estimator..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_logistic_regression.yml)
```python
properties:
  c_parameter: 0.01
  independent_vars:
    columns:
    - mean area
    - mean compactness
  normalize_cm: false
  scale: true
  solver: liblinear
  target:
    column: benign
  test_size: 0.2

```
#### Command line
```python
logistic_regression --config config_logistic_regression.yml --input_dataset_path dataset_logistic_regression.csv --output_model_path ref_output_model_logistic_regression.pkl --output_test_table_path ref_output_test_logistic_regression.csv --output_plot_path ref_output_plot_logistic_regression.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_logistic_regression.json)
```python
{
  "properties": {
    "independent_vars": {
      "columns": [
        "mean area",
        "mean compactness"
      ]
    },
    "target": {
      "column": "benign"
    },
    "solver": "liblinear",
    "c_parameter": 0.01,
    "normalize_cm": false,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
logistic_regression --config config_logistic_regression.json --input_dataset_path dataset_logistic_regression.csv --output_model_path ref_output_model_logistic_regression.pkl --output_test_table_path ref_output_test_logistic_regression.csv --output_plot_path ref_output_plot_logistic_regression.png
```

## Recurrent_neural_network
Wrapper of the TensorFlow Keras LSTM method.
### Get help
Command:
```python
recurrent_neural_network -h
```
    usage: recurrent_neural_network [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the TensorFlow Keras LSTM method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Loss, accuracy and MSE plots. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: h5.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_recurrent.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_model_recurrent.h5). Accepted formats: H5
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_recurrent.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Loss, accuracy and MSE plots. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_plot_recurrent.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **validation_size** (*number*): (0.2) Represents the proportion of the dataset to include in the validation split. It should be between 0.0 and 1.0..
* **window_size** (*integer*): (5) Number of steps for each window of training model..
* **test_size** (*integer*): (5) Represents the number of samples of the dataset to include in the test split..
* **hidden_layers** (*array*): (None) List of dictionaries with hidden layers values. Format: [ { 'size': 50, 'activation': 'relu' } ]..
* **optimizer** (*string*): (Adam) Name of optimizer instance. .
* **learning_rate** (*number*): (0.02) Determines the step size at each iteration while moving toward a minimum of a loss function.
* **batch_size** (*integer*): (100) Number of samples per gradient update..
* **max_epochs** (*integer*): (100) Number of epochs to train the model. As the early stopping is enabled, this is a maximum..
* **normalize_cm** (*boolean*): (False) Whether or not to normalize the confusion matrix..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_recurrent_neural_network.yml)
```python
properties:
  batch_size: 32
  hidden_layers:
  - activation: relu
    size: 100
  - activation: relu
    size: 50
  - activation: relu
    size: 50
  learning_rate: 0.01
  max_epochs: 50
  optimizer: Adam
  target:
    index: 1
  test_size: 12
  validation_size: 0.2
  window_size: 5

```
#### Command line
```python
recurrent_neural_network --config config_recurrent_neural_network.yml --input_dataset_path dataset_recurrent.csv --output_model_path ref_output_model_recurrent.h5 --output_test_table_path ref_output_test_recurrent.csv --output_plot_path ref_output_plot_recurrent.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_recurrent_neural_network.json)
```python
{
  "properties": {
    "target": {
      "index": 1
    },
    "window_size": 5,
    "validation_size": 0.2,
    "test_size": 12,
    "hidden_layers": [
      {
        "size": 100,
        "activation": "relu"
      },
      {
        "size": 50,
        "activation": "relu"
      },
      {
        "size": 50,
        "activation": "relu"
      }
    ],
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "batch_size": 32,
    "max_epochs": 50
  }
}
```
#### Command line
```python
recurrent_neural_network --config config_recurrent_neural_network.json --input_dataset_path dataset_recurrent.csv --output_model_path ref_output_model_recurrent.h5 --output_test_table_path ref_output_test_recurrent.csv --output_plot_path ref_output_plot_recurrent.png
```

## Linear_regression
Wrapper of the scikit-learn LinearRegression method.
### Get help
Command:
```python
linear_regression -h
```
    usage: linear_regression [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn LinearRegression method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Residual plot checks the error between actual values and predicted values. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/dataset_linear_regression.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_model_linear_regression.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_test_linear_regression.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Residual plot checks the error between actual values and predicted values. Sample file. File type: output. [Sample file](None). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_linear_regression.yml)
```python
properties:
  independent_vars:
    columns:
    - size
    - year
    - view
  scale: true
  target:
    column: price
  test_size: 0.2

```
#### Command line
```python
linear_regression --config config_linear_regression.yml --input_dataset_path dataset_linear_regression.csv --output_model_path ref_output_model_linear_regression.pkl --output_test_table_path ref_output_test_linear_regression.csv --output_plot_path output.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_linear_regression.json)
```python
{
  "properties": {
    "independent_vars": {
      "columns": [
        "size",
        "year",
        "view"
      ]
    },
    "target": {
      "column": "price"
    },
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
linear_regression --config config_linear_regression.json --input_dataset_path dataset_linear_regression.csv --output_model_path ref_output_model_linear_regression.pkl --output_test_table_path ref_output_test_linear_regression.csv --output_plot_path output.png
```

## Random_forest_classifier
Wrapper of the scikit-learn RandomForestClassifier method.
### Get help
Command:
```python
random_forest_classifier -h
```
    usage: random_forest_classifier [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn RandomForestClassifier method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_random_forest_classifier.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_random_forest_classifier.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_random_forest_classifier.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_random_forest_classifier.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **n_estimators** (*integer*): (100) The number of trees in the forest..
* **bootstrap** (*boolean*): (True) Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree..
* **normalize_cm** (*boolean*): (False) Whether or not to normalize the confusion matrix..
* **random_state_method** (*integer*): (5) Controls the randomness of the estimator..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_random_forest_classifier.yml)
```python
properties:
  bootstrap: true
  independent_vars:
    indexes:
    - 0
    - 1
    - 2
    - 3
    - 4
  n_estimators: 100
  normalize_cm: false
  scale: true
  target:
    index: 5
  test_size: 0.2

```
#### Command line
```python
random_forest_classifier --config config_random_forest_classifier.yml --input_dataset_path dataset_random_forest_classifier.csv --output_model_path ref_output_model_random_forest_classifier.pkl --output_test_table_path ref_output_test_random_forest_classifier.csv --output_plot_path ref_output_plot_random_forest_classifier.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_random_forest_classifier.json)
```python
{
  "properties": {
    "independent_vars": {
      "indexes": [
        0,
        1,
        2,
        3,
        4
      ]
    },
    "target": {
      "index": 5
    },
    "n_estimators": 100,
    "bootstrap": true,
    "normalize_cm": false,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
random_forest_classifier --config config_random_forest_classifier.json --input_dataset_path dataset_random_forest_classifier.csv --output_model_path ref_output_model_random_forest_classifier.pkl --output_test_table_path ref_output_test_random_forest_classifier.csv --output_plot_path ref_output_plot_random_forest_classifier.png
```

## K_means_coefficient
Wrapper of the scikit-learn KMeans method.
### Get help
Command:
```python
k_means_coefficient -h
```
    usage: k_means_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn KMeans method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the elbow and gap methods plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_k_means_coefficient.csv). Accepted formats: CSV
* **output_results_path** (*string*): Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_k_means_coefficient.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the elbow method and gap statistics plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_k_means_coefficient.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictors** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **max_clusters** (*integer*): (6) Maximum number of clusters to use by default for kmeans queries..
* **random_state_method** (*integer*): (5) Determines random number generation for centroid initialization..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_means_coefficient.yml)
```python
properties:
  max_clusters: 10
  predictors:
    columns:
    - sepal_length
    - sepal_width
  scale: true

```
#### Command line
```python
k_means_coefficient --config config_k_means_coefficient.yml --input_dataset_path dataset_k_means_coefficient.csv --output_results_path ref_output_results_k_means_coefficient.csv --output_plot_path ref_output_plot_k_means_coefficient.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_means_coefficient.json)
```python
{
  "properties": {
    "predictors": {
      "columns": [
        "sepal_length",
        "sepal_width"
      ]
    },
    "max_clusters": 10,
    "scale": true
  }
}
```
#### Command line
```python
k_means_coefficient --config config_k_means_coefficient.json --input_dataset_path dataset_k_means_coefficient.csv --output_results_path ref_output_results_k_means_coefficient.csv --output_plot_path ref_output_plot_k_means_coefficient.png
```

## Pls_regression
Wrapper of the scikit-learn PLSRegression method.
### Get help
Command:
```python
pls_regression -h
```
    usage: pls_regression [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn PLSRegression method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the R2 cross-validation plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Table with R2 and MSE for calibration and cross-validation data. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/dimensionality_reduction/dataset_pls_regression.csv). Accepted formats: CSV
* **output_results_path** (*string*): Table with R2 and MSE for calibration and cross-validation data. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_results_pls_regression.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the R2 cross-validation plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_plot_pls_regression.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **n_components** (*integer*): (5) Maximum number of components to use by default for PLS queries..
* **cv** (*integer*): (10) Specify the number of folds in the cross-validation splitting strategy. Value must be betwwen 2 and number of samples in the dataset..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_pls_regression.yml)
```python
properties:
  cv: 10
  features:
    range:
    - - 0
      - 29
  n_components: 12
  scale: true
  target:
    index: 30

```
#### Command line
```python
pls_regression --config config_pls_regression.yml --input_dataset_path dataset_pls_regression.csv --output_results_path ref_output_results_pls_regression.csv --output_plot_path ref_output_plot_pls_regression.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_pls_regression.json)
```python
{
  "properties": {
    "features": {
      "range": [
        [
          0,
          29
        ]
      ]
    },
    "target": {
      "index": 30
    },
    "n_components": 12,
    "cv": 10,
    "scale": true
  }
}
```
#### Command line
```python
pls_regression --config config_pls_regression.json --input_dataset_path dataset_pls_regression.csv --output_results_path ref_output_results_pls_regression.csv --output_plot_path ref_output_plot_pls_regression.png
```

## Regression_neural_network
Wrapper of the TensorFlow Keras Sequential method.
### Get help
Command:
```python
regression_neural_network -h
```
    usage: regression_neural_network [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the TensorFlow Keras Sequential method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Loss, MAE and MSE plots. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: h5.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_regression.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](http://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_model_regression.h5). Accepted formats: H5
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_regression.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Loss, MAE and MSE plots. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_plot_regression.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Independent variables or columns from your dataset you want to train. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **validation_size** (*number*): (0.2) Represents the proportion of the dataset to include in the validation split. It should be between 0.0 and 1.0..
* **test_size** (*number*): (0.1) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **hidden_layers** (*array*): (None) List of dictionaries with hidden layers values. Format: [ { 'size': 50, 'activation': 'relu' } ]..
* **output_layer_activation** (*string*): (softmax) Activation function to use in the output layer. .
* **optimizer** (*string*): (Adam) Name of optimizer instance. .
* **learning_rate** (*number*): (0.02) Determines the step size at each iteration while moving toward a minimum of a loss function.
* **batch_size** (*integer*): (100) Number of samples per gradient update..
* **max_epochs** (*integer*): (100) Number of epochs to train the model. As the early stopping is enabled, this is a maximum..
* **random_state** (*integer*): (5) Controls the shuffling applied to the data before applying the split. ..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_regression_neural_network.yml)
```python
properties:
  batch_size: 32
  features:
    columns:
    - ZN
    - RM
    - AGE
    - LSTAT
  hidden_layers:
  - activation: relu
    size: 10
  - activation: relu
    size: 8
  learning_rate: 0.01
  max_epochs: 150
  optimizer: Adam
  target:
    column: MEDV
  test_size: 0.2
  validation_size: 0.2

```
#### Command line
```python
regression_neural_network --config config_regression_neural_network.yml --input_dataset_path dataset_regression.csv --output_model_path ref_output_model_regression.h5 --output_test_table_path ref_output_test_regression.csv --output_plot_path ref_output_plot_regression.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_regression_neural_network.json)
```python
{
  "properties": {
    "features": {
      "columns": [
        "ZN",
        "RM",
        "AGE",
        "LSTAT"
      ]
    },
    "target": {
      "column": "MEDV"
    },
    "validation_size": 0.2,
    "test_size": 0.2,
    "hidden_layers": [
      {
        "size": 10,
        "activation": "relu"
      },
      {
        "size": 8,
        "activation": "relu"
      }
    ],
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "batch_size": 32,
    "max_epochs": 150
  }
}
```
#### Command line
```python
regression_neural_network --config config_regression_neural_network.json --input_dataset_path dataset_regression.csv --output_model_path ref_output_model_regression.h5 --output_test_table_path ref_output_test_regression.csv --output_plot_path ref_output_plot_regression.png
```

## Autoencoder_neural_network
Wrapper of the TensorFlow Keras LSTM method for encoding.
### Get help
Command:
```python
autoencoder_neural_network -h
```
    usage: autoencoder_neural_network [-h] [--config CONFIG] --input_decode_path INPUT_DECODE_PATH [--input_predict_path INPUT_PREDICT_PATH] --output_model_path OUTPUT_MODEL_PATH [--output_test_decode_path OUTPUT_TEST_DECODE_PATH] [--output_test_predict_path OUTPUT_TEST_PREDICT_PATH]
    
    Wrapper of the TensorFlow Keras LSTM method for encoding.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --input_predict_path INPUT_PREDICT_PATH
                            Path to the input predict dataset. Accepted formats: csv.
      --output_test_decode_path OUTPUT_TEST_DECODE_PATH
                            Path to the test decode table file. Accepted formats: csv.
      --output_test_predict_path OUTPUT_TEST_PREDICT_PATH
                            Path to the test predict table file. Accepted formats: csv.
    
    required arguments:
      --input_decode_path INPUT_DECODE_PATH
                            Path to the input decode dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: h5.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_decode_path** (*string*): Path to the input decode dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_autoencoder_decode.csv). Accepted formats: CSV
* **input_predict_path** (*string*): Path to the input predict dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_autoencoder_predict.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_model_autoencoder.h5). Accepted formats: H5
* **output_test_decode_path** (*string*): Path to the test decode table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_decode_autoencoder.csv). Accepted formats: CSV
* **output_test_predict_path** (*string*): Path to the test predict table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_predict_autoencoder.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **optimizer** (*string*): (Adam) Name of optimizer instance. .
* **learning_rate** (*number*): (0.02) Determines the step size at each iteration while moving toward a minimum of a loss function.
* **batch_size** (*integer*): (100) Number of samples per gradient update..
* **max_epochs** (*integer*): (100) Number of epochs to train the model. As the early stopping is enabled, this is a maximum..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_autoencoder_neural_network.yml)
```python
properties:
  batch_size: 32
  learning_rate: 0.01
  max_epochs: 300
  optimizer: Adam

```
#### Command line
```python
autoencoder_neural_network --config config_autoencoder_neural_network.yml --input_decode_path dataset_autoencoder_decode.csv --input_predict_path dataset_autoencoder_predict.csv --output_model_path ref_output_model_autoencoder.h5 --output_test_decode_path ref_output_test_decode_autoencoder.csv --output_test_predict_path ref_output_test_predict_autoencoder.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_autoencoder_neural_network.json)
```python
{
  "properties": {
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "batch_size": 32,
    "max_epochs": 300
  }
}
```
#### Command line
```python
autoencoder_neural_network --config config_autoencoder_neural_network.json --input_decode_path dataset_autoencoder_decode.csv --input_predict_path dataset_autoencoder_predict.csv --output_model_path ref_output_model_autoencoder.h5 --output_test_decode_path ref_output_test_decode_autoencoder.csv --output_test_predict_path ref_output_test_predict_autoencoder.csv
```

## Spectral_coefficient
Wrapper of the scikit-learn SpectralClustering method.
### Get help
Command:
```python
spectral_coefficient -h
```
    usage: spectral_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn SpectralClustering method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the elbow and gap methods plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_spectral_coefficient.csv). Accepted formats: CSV
* **output_results_path** (*string*): Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_spectral_coefficient.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the elbow method and gap statistics plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_spectral_coefficient.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictors** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **max_clusters** (*integer*): (6) Maximum number of clusters to use by default for kmeans queries..
* **random_state_method** (*integer*): (5) A pseudo random number generator used for the initialization of the lobpcg eigen vectors decomposition when *eigen_solver='amg'* and by the K-Means initialization..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_spectral_coefficient.yml)
```python
properties:
  max_clusters: 10
  predictors:
    columns:
    - sepal_length
    - sepal_width
  scale: true

```
#### Command line
```python
spectral_coefficient --config config_spectral_coefficient.yml --input_dataset_path dataset_spectral_coefficient.csv --output_results_path ref_output_results_spectral_coefficient.csv --output_plot_path ref_output_plot_spectral_coefficient.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_spectral_coefficient.json)
```python
{
  "properties": {
    "predictors": {
      "columns": [
        "sepal_length",
        "sepal_width"
      ]
    },
    "max_clusters": 10,
    "scale": true
  }
}
```
#### Command line
```python
spectral_coefficient --config config_spectral_coefficient.json --input_dataset_path dataset_spectral_coefficient.csv --output_results_path ref_output_results_spectral_coefficient.csv --output_plot_path ref_output_plot_spectral_coefficient.png
```

## Drop_columns
Drops columns from a given dataset.
### Get help
Command:
```python
drop_columns -h
```
    usage: drop_columns [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_dataset_path OUTPUT_DATASET_PATH
    
    Drops columns from a given dataset.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_dataset_path OUTPUT_DATASET_PATH
                            Path to the output dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_drop.csv). Accepted formats: CSV
* **output_dataset_path** (*string*): Path to the output dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_drop.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **targets** (*object*): ({}) Independent variables or columns from your dataset you want to drop. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_drop_columns.yml)
```python
properties:
  targets:
    columns:
    - WEIGHT
    - SCORE

```
#### Command line
```python
drop_columns --config config_drop_columns.yml --input_dataset_path dataset_drop.csv --output_dataset_path ref_output_drop.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_drop_columns.json)
```python
{
  "properties": {
    "targets": {
      "columns": [
        "WEIGHT",
        "SCORE"
      ]
    }
  }
}
```
#### Command line
```python
drop_columns --config config_drop_columns.json --input_dataset_path dataset_drop.csv --output_dataset_path ref_output_drop.csv
```

## Decision_tree
Wrapper of the scikit-learn DecisionTreeClassifier method.
### Get help
Command:
```python
decision_tree -h
```
    usage: decision_tree [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn DecisionTreeClassifier method. 
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_decision_tree.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_decision_tree.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_decision_tree.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_decision_tree.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **criterion** (*string*): (gini) The function to measure the quality of a split. .
* **max_depth** (*integer*): (4) The maximum depth of the model. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples..
* **normalize_cm** (*boolean*): (False) Whether or not to normalize the confusion matrix..
* **random_state_method** (*integer*): (5) Controls the randomness of the estimator..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_decision_tree.yml)
```python
properties:
  criterion: entropy
  independent_vars:
    columns:
    - interest_rate
    - credit
    - march
    - previous
    - duration
  max_depth: 4
  normalize_cm: false
  scale: true
  target:
    column: y
  test_size: 0.2

```
#### Command line
```python
decision_tree --config config_decision_tree.yml --input_dataset_path dataset_decision_tree.csv --output_model_path ref_output_model_decision_tree.pkl --output_test_table_path ref_output_test_decision_tree.csv --output_plot_path ref_output_plot_decision_tree.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_decision_tree.json)
```python
{
  "properties": {
    "independent_vars": {
      "columns": [
        "interest_rate",
        "credit",
        "march",
        "previous",
        "duration"
      ]
    },
    "target": {
      "column": "y"
    },
    "criterion": "entropy",
    "max_depth": 4,
    "normalize_cm": false,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
decision_tree --config config_decision_tree.json --input_dataset_path dataset_decision_tree.csv --output_model_path ref_output_model_decision_tree.pkl --output_test_table_path ref_output_test_decision_tree.csv --output_plot_path ref_output_plot_decision_tree.png
```

## Undersampling
Wrapper of most of the imblearn.under_sampling methods.
### Get help
Command:
```python
undersampling -h
```
    usage: undersampling [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_dataset_path OUTPUT_DATASET_PATH
    
    Wrapper of most of the imblearn.under_sampling methods.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_dataset_path OUTPUT_DATASET_PATH
                            Path to the output dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/resampling/dataset_resampling.csv). Accepted formats: CSV
* **output_dataset_path** (*string*): Path to the output dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/resampling/ref_output_undersampling.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **method** (*string*): (None) Undersampling method. It's a mandatory property. .
* **type** (*string*): (None) Type of oversampling. It's a mandatory property. .
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **evaluate** (*boolean*): (False) Whether or not to evaluate the dataset before and after applying the resampling..
* **evaluate_splits** (*integer*): (3) Number of folds to be applied by the Repeated Stratified K-Fold evaluation method. Must be at least 2..
* **evaluate_repeats** (*integer*): (3) Number of times Repeated Stratified K-Fold cross validator needs to be repeated..
* **n_bins** (*integer*): (5) Only for regression undersampling. The number of classes that the user wants to generate with the target data..
* **balanced_binning** (*boolean*): (False) Only for regression undersampling. Decides whether samples are to be distributed roughly equally across all classes..
* **sampling_strategy** (*object*): ({'target': 'auto'}) Sampling information to sample the data set. Formats: { "target": "auto" }, { "ratio": 0.3 }, { "dict": { 0: 300, 1: 200, 2: 100 } } or { "list": [0, 2, 3] }. When "target", specify the class targeted by the resampling; the number of samples in the different classes will be equalized; possible choices are: majority (resample only the majority class), not minority (resample all classes but the minority class), not majority (resample all classes but the majority class), all (resample all classes), auto (equivalent to 'not minority'). When "ratio", it corresponds to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling (ONLY IN CASE OF BINARY CLASSIFICATION). When "dict", the keys correspond to the targeted classes, the values correspond to the desired number of samples for each targeted class. When "list", the list contains the classes targeted by the resampling..
* **version** (*integer*): (1) Only for NearMiss method. Version of the NearMiss to use. .
* **n_neighbors** (*integer*): (1) Only for NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours and NeighbourhoodCleaningRule methods. Size of the neighbourhood to consider to compute the average distance to the minority point samples..
* **threshold_cleaning** (*number*): (0.5) Only for NeighbourhoodCleaningRule method. Threshold used to whether consider a class or not during the cleaning after applying ENN..
* **random_state_method** (*integer*): (5) Only for RandomUnderSampler and ClusterCentroids methods. Controls the randomization of the algorithm..
* **random_state_evaluate** (*integer*): (5) Controls the shuffling applied to the Repeated Stratified K-Fold evaluation method..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_undersampling.yml)
```python
properties:
  evaluate: true
  method: enn
  n_bins: 10
  n_neighbors: 3
  target:
    column: VALUE
  type: regression

```
#### Command line
```python
undersampling --config config_undersampling.yml --input_dataset_path dataset_resampling.csv --output_dataset_path ref_output_undersampling.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_undersampling.json)
```python
{
  "properties": {
    "method": "enn",
    "type": "regression",
    "target": {
      "column": "VALUE"
    },
    "evaluate": true,
    "n_bins": 10,
    "n_neighbors": 3
  }
}
```
#### Command line
```python
undersampling --config config_undersampling.json --input_dataset_path dataset_resampling.csv --output_dataset_path ref_output_undersampling.csv
```

## Pairwise_comparison
Generates a pairwise comparison from a given dataset.
### Get help
Command:
```python
pairwise_comparison -h
```
    usage: pairwise_comparison [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_plot_path OUTPUT_PLOT_PATH
    
    Generates a pairwise comparison from a given dataset
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the pairwise comparison plot. Accepted formats: png.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_pairwise_comparison.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the pairwise comparison plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_plot_pairwise_comparison.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Independent variables or columns from your dataset you want to compare. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_pairwise_comparison.yml)
```python
properties:
  features:
    indexes:
    - 0
    - 1
    - 2
    - 3

```
#### Command line
```python
pairwise_comparison --config config_pairwise_comparison.yml --input_dataset_path dataset_pairwise_comparison.csv --output_plot_path ref_output_plot_pairwise_comparison.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_pairwise_comparison.json)
```python
{
  "properties": {
    "features": {
      "indexes": [
        0,
        1,
        2,
        3
      ]
    }
  }
}
```
#### Command line
```python
pairwise_comparison --config config_pairwise_comparison.json --input_dataset_path dataset_pairwise_comparison.csv --output_plot_path ref_output_plot_pairwise_comparison.png
```

## Pls_components
Wrapper of the scikit-learn PLSRegression method.
### Get help
Command:
```python
pls_components -h
```
    usage: pls_components [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn PLSRegression method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the Mean Square Error plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Table with R2 and MSE for calibration and cross-validation data for the best number of components. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/dimensionality_reduction/dataset_pls_components.csv). Accepted formats: CSV
* **output_results_path** (*string*): Table with R2 and MSE for calibration and cross-validation data for the best number of components. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_results_pls_components.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the Mean Square Error plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/dimensionality_reduction/ref_output_plot_pls_components.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **optimise** (*boolean*): (False) Whether or not optimise the process of MSE calculation. Beware, if True selected, the process can take a long time depending on the **max_components** value..
* **max_components** (*integer*): (10) Maximum number of components to use by default for PLS queries..
* **cv** (*integer*): (10) Specify the number of folds in the cross-validation splitting strategy. Value must be between 2 and number of samples in the dataset..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_pls_components.yml)
```python
properties:
  cv: 10
  features:
    range:
    - - 0
      - 29
  max_components: 30
  optimise: false
  scale: true
  target:
    index: 30

```
#### Command line
```python
pls_components --config config_pls_components.yml --input_dataset_path dataset_pls_components.csv --output_results_path ref_output_results_pls_components.csv --output_plot_path ref_output_plot_pls_components.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_pls_components.json)
```python
{
  "properties": {
    "features": {
      "range": [
        [
          0,
          29
        ]
      ]
    },
    "target": {
      "index": 30
    },
    "optimise": false,
    "max_components": 30,
    "cv": 10,
    "scale": true
  }
}
```
#### Command line
```python
pls_components --config config_pls_components.json --input_dataset_path dataset_pls_components.csv --output_results_path ref_output_results_pls_components.csv --output_plot_path ref_output_plot_pls_components.png
```

## Clustering_predict
Makes predictions from an input dataset and a given clustering model.
### Get help
Command:
```python
clustering_predict -h
```
    usage: clustering_predict [-h] [--config CONFIG] --input_model_path INPUT_MODEL_PATH --output_results_path OUTPUT_RESULTS_PATH [--input_dataset_path INPUT_DATASET_PATH]
    
    Makes predictions from an input dataset and a given clustering model.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the dataset to predict. Accepted formats: csv.
    
    required arguments:
      --input_model_path INPUT_MODEL_PATH
                            Path to the input model. Accepted formats: pkl.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the output results file. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_path** (*string*): Path to the input model. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/model_clustering_predict.pkl). Accepted formats: PKL
* **input_dataset_path** (*string*): Path to the dataset to predict. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/input_clustering_predict.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the output results file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_clustering_predict.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictions** (*array*): (None) List of dictionaries with all values you want to predict targets. It will be taken into account only in case **input_dataset_path** is not provided. Format: [{ 'var1': 1.0, 'var2': 2.0 }, { 'var1': 4.0, 'var2': 2.7 }] for datasets with headers and [[ 1.0, 2.0 ], [ 4.0, 2.7 ]] for datasets without headers..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_clustering_predict.yml)
```python
properties:
  predictions:
  - petal_length: 1.4
    petal_width: 0.2
    sepal_length: 5.1
    sepal_width: 3.5
  - petal_length: 5.2
    petal_width: 2.3
    sepal_length: 6.7
    sepal_width: 3.0
  - petal_length: 5.0
    petal_width: 1.9
    sepal_length: 6.3
    sepal_width: 2.5

```
#### Command line
```python
clustering_predict --config config_clustering_predict.yml --input_model_path model_clustering_predict.pkl --input_dataset_path input_clustering_predict.csv --output_results_path ref_output_results_clustering_predict.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_clustering_predict.json)
```python
{
  "properties": {
    "predictions": [
      {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      },
      {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3
      },
      {
        "sepal_length": 6.3,
        "sepal_width": 2.5,
        "petal_length": 5.0,
        "petal_width": 1.9
      }
    ]
  }
}
```
#### Command line
```python
clustering_predict --config config_clustering_predict.json --input_model_path model_clustering_predict.pkl --input_dataset_path input_clustering_predict.csv --output_results_path ref_output_results_clustering_predict.csv
```

## Agglomerative_coefficient
Wrapper of the scikit-learn AgglomerativeCoefficient method.
### Get help
Command:
```python
agglomerative_coefficient -h
```
    usage: agglomerative_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn AgglomerativeCoefficient method. 
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the elbow and gap methods plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the gap values list. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_agglomerative_coefficient.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the gap values list. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_agglomerative_coefficient.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the elbow method and gap statistics plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_agglomerative_coefficient.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictors** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **max_clusters** (*integer*): (6) Maximum number of clusters to use by default for kmeans queries..
* **affinity** (*string*): (euclidean) Metric used to compute the linkage. If linkage is "ward", only "euclidean" is accepted. .
* **linkage** (*string*): (ward) The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. .
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_agglomerative_coefficient.yml)
```python
properties:
  max_clusters: 10
  predictors:
    columns:
    - sepal_length
    - sepal_width
  scale: true

```
#### Command line
```python
agglomerative_coefficient --config config_agglomerative_coefficient.yml --input_dataset_path dataset_agglomerative_coefficient.csv --output_results_path ref_output_results_agglomerative_coefficient.csv --output_plot_path ref_output_plot_agglomerative_coefficient.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_agglomerative_coefficient.json)
```python
{
  "properties": {
    "predictors": {
      "columns": [
        "sepal_length",
        "sepal_width"
      ]
    },
    "max_clusters": 10,
    "scale": true
  }
}
```
#### Command line
```python
agglomerative_coefficient --config config_agglomerative_coefficient.json --input_dataset_path dataset_agglomerative_coefficient.csv --output_results_path ref_output_results_agglomerative_coefficient.csv --output_plot_path ref_output_plot_agglomerative_coefficient.png
```

## Spectral_clustering
Wrapper of the scikit-learn SpectralClustering method.
### Get help
Command:
```python
spectral_clustering -h
```
    usage: spectral_clustering [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn SpectralClustering method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the clustering plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the clustered dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_spectral_clustering.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_spectral_clustering.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_spectral_clustering.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictors** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **clusters** (*integer*): (3) The number of clusters to form as well as the number of centroids to generate..
* **affinity** (*string*): (rbf) How to construct the affinity matrix. .
* **plots** (*array*): (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ]..
* **random_state_method** (*integer*): (5) A pseudo random number generator used for the initialization of the lobpcg eigen vectors decomposition when *eigen_solver='amg'* and by the K-Means initialization..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_spectral_clustering.yml)
```python
properties:
  affinity: nearest_neighbors
  clusters: 3
  plots:
  - features:
    - sepal_length
    - sepal_width
    title: Plot 1
  - features:
    - petal_length
    - petal_width
    title: Plot 2
  - features:
    - sepal_length
    - sepal_width
    - petal_length
    title: Plot 3
  - features:
    - petal_length
    - petal_width
    - sepal_width
    title: Plot 4
  - features:
    - sepal_length
    - petal_width
    title: Plot 5
  predictors:
    columns:
    - sepal_length
    - sepal_width
    - petal_length
    - petal_width
  scale: true

```
#### Command line
```python
spectral_clustering --config config_spectral_clustering.yml --input_dataset_path dataset_spectral_clustering.csv --output_results_path ref_output_results_spectral_clustering.csv --output_plot_path ref_output_plot_spectral_clustering.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_spectral_clustering.json)
```python
{
  "properties": {
    "predictors": {
      "columns": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
      ]
    },
    "clusters": 3,
    "affinity": "nearest_neighbors",
    "plots": [
      {
        "title": "Plot 1",
        "features": [
          "sepal_length",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 2",
        "features": [
          "petal_length",
          "petal_width"
        ]
      },
      {
        "title": "Plot 3",
        "features": [
          "sepal_length",
          "sepal_width",
          "petal_length"
        ]
      },
      {
        "title": "Plot 4",
        "features": [
          "petal_length",
          "petal_width",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 5",
        "features": [
          "sepal_length",
          "petal_width"
        ]
      }
    ],
    "scale": true
  }
}
```
#### Command line
```python
spectral_clustering --config config_spectral_clustering.json --input_dataset_path dataset_spectral_clustering.csv --output_results_path ref_output_results_spectral_clustering.csv --output_plot_path ref_output_plot_spectral_clustering.png
```

## K_neighbors_coefficient
Wrapper of the scikit-learn KNeighborsClassifier method.
### Get help
Command:
```python
k_neighbors_coefficient -h
```
    usage: k_neighbors_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn KNeighborsClassifier method. 
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the accuracy plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the accuracy values list. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_k_neighbors_coefficient.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the accuracy values list. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_k_neighbors_coefficient.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the accuracy plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_k_neighbors_coefficient.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*array*): (None) Independent variables or columns from your dataset you want to train..
* **target** (*string*): (None) Dependent variable or column from your dataset you want to predict..
* **metric** (*string*): (minkowski) The distance metric to use for the tree. .
* **max_neighbors** (*integer*): (6) Maximum number of neighbors to use by default for kneighbors queries..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_neighbors_coefficient.yml)
```python
properties:
  independent_vars:
    columns:
    - region
    - tenure
    - age
    - marital
    - address
    - income
    - ed
    - employ
    - retire
    - gender
    - reside
  max_neighbors: 15
  metric: minkowski
  scale: true
  target:
    column: custcat
  test_size: 0.2

```
#### Command line
```python
k_neighbors_coefficient --config config_k_neighbors_coefficient.yml --input_dataset_path dataset_k_neighbors_coefficient.csv --output_results_path ref_output_test_k_neighbors_coefficient.csv --output_plot_path ref_output_plot_k_neighbors_coefficient.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_k_neighbors_coefficient.json)
```python
{
  "properties": {
    "independent_vars": {
      "columns": [
        "region",
        "tenure",
        "age",
        "marital",
        "address",
        "income",
        "ed",
        "employ",
        "retire",
        "gender",
        "reside"
      ]
    },
    "target": {
      "column": "custcat"
    },
    "metric": "minkowski",
    "max_neighbors": 15,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
k_neighbors_coefficient --config config_k_neighbors_coefficient.json --input_dataset_path dataset_k_neighbors_coefficient.csv --output_results_path ref_output_test_k_neighbors_coefficient.csv --output_plot_path ref_output_plot_k_neighbors_coefficient.png
```

## Classification_neural_network
Wrapper of the TensorFlow Keras Sequential method.
### Get help
Command:
```python
classification_neural_network -h
```
    usage: classification_neural_network [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the TensorFlow Keras Sequential method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Loss, accuracy and MSE plots. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: h5.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_classification.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_model_classification.h5). Accepted formats: H5
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_test_classification.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Loss, accuracy and MSE plots. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_plot_classification.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Independent variables or columns from your dataset you want to train. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **validation_size** (*number*): (0.2) Represents the proportion of the dataset to include in the validation split. It should be between 0.0 and 1.0..
* **test_size** (*number*): (0.1) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **hidden_layers** (*array*): (None) List of dictionaries with hidden layers values. Format: [ { 'size': 50, 'activation': 'relu' } ]..
* **output_layer_activation** (*string*): (softmax) Activation function to use in the output layer. .
* **optimizer** (*string*): (Adam) Name of optimizer instance. .
* **learning_rate** (*number*): (0.02) Determines the step size at each iteration while moving toward a minimum of a loss function.
* **batch_size** (*integer*): (100) Number of samples per gradient update..
* **max_epochs** (*integer*): (100) Number of epochs to train the model. As the early stopping is enabled, this is a maximum..
* **normalize_cm** (*boolean*): (False) Whether or not to normalize the confusion matrix..
* **random_state** (*integer*): (5) Controls the shuffling applied to the data before applying the split. ..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_classification_neural_network.yml)
```python
properties:
  batch_size: 100
  features:
    columns:
    - mean radius
    - mean texture
    - mean perimeter
    - mean area
    - mean smoothness
    - mean compactness
    - mean concavity
    - mean concave points
    - mean symmetry
    - mean fractal dimension
    - radius error
    - texture error
    - perimeter error
    - area error
    - smoothness error
    - compactness error
    - concavity error
    - concave points error
    - symmetry error
    - fractal dimension error
    - worst radius
    - worst texture
    - worst perimeter
    - worst area
    - worst smoothness
    - worst compactness
    - worst concavity
    - worst concave points
    - worst symmetry
    - worst fractal dimension
  hidden_layers:
  - activation: relu
    size: 50
  - activation: relu
    size: 50
  learning_rate: 0.02
  max_epochs: 100
  optimizer: Adam
  output_layer_activation: softmax
  scale: true
  target:
    column: benign
  test_size: 0.1
  validation_size: 0.2

```
#### Command line
```python
classification_neural_network --config config_classification_neural_network.yml --input_dataset_path dataset_classification.csv --output_model_path ref_output_model_classification.h5 --output_test_table_path ref_output_test_classification.csv --output_plot_path ref_output_plot_classification.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_classification_neural_network.json)
```python
{
  "properties": {
    "features": {
      "columns": [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension",
        "radius error",
        "texture error",
        "perimeter error",
        "area error",
        "smoothness error",
        "compactness error",
        "concavity error",
        "concave points error",
        "symmetry error",
        "fractal dimension error",
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
        "worst smoothness",
        "worst compactness",
        "worst concavity",
        "worst concave points",
        "worst symmetry",
        "worst fractal dimension"
      ]
    },
    "target": {
      "column": "benign"
    },
    "validation_size": 0.2,
    "test_size": 0.1,
    "hidden_layers": [
      {
        "size": 50,
        "activation": "relu"
      },
      {
        "size": 50,
        "activation": "relu"
      }
    ],
    "output_layer_activation": "softmax",
    "optimizer": "Adam",
    "learning_rate": 0.02,
    "batch_size": 100,
    "max_epochs": 100,
    "scale": true
  }
}
```
#### Command line
```python
classification_neural_network --config config_classification_neural_network.json --input_dataset_path dataset_classification.csv --output_model_path ref_output_model_classification.h5 --output_test_table_path ref_output_test_classification.csv --output_plot_path ref_output_plot_classification.png
```

## Classification_predict
Makes predictions from an input dataset and a given classification model.
### Get help
Command:
```python
classification_predict -h
```
    usage: classification_predict [-h] [--config CONFIG] --input_model_path INPUT_MODEL_PATH --output_results_path OUTPUT_RESULTS_PATH [--input_dataset_path INPUT_DATASET_PATH]
    
    Makes predictions from an input dataset and a given classification model.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the dataset to predict. Accepted formats: csv.
    
    required arguments:
      --input_model_path INPUT_MODEL_PATH
                            Path to the input model. Accepted formats: pkl.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the output results file. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_path** (*string*): Path to the input model. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/model_classification_predict.pkl). Accepted formats: PKL
* **input_dataset_path** (*string*): Path to the dataset to predict. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/input_classification_predict.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the output results file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_classification_predict.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictions** (*array*): (None) List of dictionaries with all values you want to predict targets. It will be taken into account only in case **input_dataset_path** is not provided. Format: [{ 'var1': 1.0, 'var2': 2.0 }, { 'var1': 4.0, 'var2': 2.7 }] for datasets with headers and [[ 1.0, 2.0 ], [ 4.0, 2.7 ]] for datasets without headers..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_classification_predict.yml)
```python
properties:
  remove_tmp: false

```
#### Command line
```python
classification_predict --config config_classification_predict.yml --input_model_path model_classification_predict.pkl --input_dataset_path input_classification_predict.csv --output_results_path ref_output_classification_predict.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_classification_predict.json)
```python
{
  "properties": {
    "remove_tmp": false
  }
}
```
#### Command line
```python
classification_predict --config config_classification_predict.json --input_model_path model_classification_predict.pkl --input_dataset_path input_classification_predict.csv --output_results_path ref_output_classification_predict.csv
```

## Dummy_variables
Converts categorical variables into dummy/indicator variables (binaries).
### Get help
Command:
```python
dummy_variables -h
```
    usage: dummy_variables [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_dataset_path OUTPUT_DATASET_PATH
    
    Maps dummy variables from a given dataset.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_dataset_path OUTPUT_DATASET_PATH
                            Path to the output dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_dummy_variables.csv). Accepted formats: CSV
* **output_dataset_path** (*string*): Path to the output dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_dataset_dummy_variables.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **targets** (*object*): ({}) Independent variables or columns from your dataset you want to drop. If None given, all the columns will be taken. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_dummy_variables.yml)
```python
properties:
  targets:
    columns:
    - view

```
#### Command line
```python
dummy_variables --config config_dummy_variables.yml --input_dataset_path dataset_dummy_variables.csv --output_dataset_path ref_output_dataset_dummy_variables.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_dummy_variables.json)
```python
{
  "properties": {
    "targets": {
      "columns": [
        "view"
      ]
    }
  }
}
```
#### Command line
```python
dummy_variables --config config_dummy_variables.json --input_dataset_path dataset_dummy_variables.csv --output_dataset_path ref_output_dataset_dummy_variables.csv
```

## Agglomerative_clustering
Wrapper of the scikit-learn AgglomerativeClustering method.
### Get help
Command:
```python
agglomerative_clustering -h
```
    usage: agglomerative_clustering [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn AgglomerativeClustering method. 
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the clustering plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the clustered dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_agglomerative_clustering.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_agglomerative_clustering.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_agglomerative_clustering.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictors** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of multiple formats, the first one will be picked..
* **clusters** (*integer*): (3) The number of clusters to form as well as the number of centroids to generate..
* **affinity** (*string*): (euclidean) Metric used to compute the linkage. If linkage is "ward", only "euclidean" is accepted. .
* **linkage** (*string*): (ward) The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. .
* **plots** (*array*): (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ]..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_agglomerative_clustering.yml)
```python
properties:
  clusters: 3
  linkage: average
  plots:
  - features:
    - sepal_length
    - sepal_width
    title: Plot 1
  - features:
    - petal_length
    - petal_width
    title: Plot 2
  - features:
    - sepal_length
    - sepal_width
    - petal_length
    title: Plot 3
  - features:
    - petal_length
    - petal_width
    - sepal_width
    title: Plot 4
  - features:
    - sepal_length
    - petal_width
    title: Plot 5
  predictors:
    columns:
    - sepal_length
    - sepal_width
    - petal_length
    - petal_width
  scale: true

```
#### Command line
```python
agglomerative_clustering --config config_agglomerative_clustering.yml --input_dataset_path dataset_agglomerative_clustering.csv --output_results_path ref_output_results_agglomerative_clustering.csv --output_plot_path ref_output_plot_agglomerative_clustering.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_agglomerative_clustering.json)
```python
{
  "properties": {
    "predictors": {
      "columns": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
      ]
    },
    "clusters": 3,
    "linkage": "average",
    "plots": [
      {
        "title": "Plot 1",
        "features": [
          "sepal_length",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 2",
        "features": [
          "petal_length",
          "petal_width"
        ]
      },
      {
        "title": "Plot 3",
        "features": [
          "sepal_length",
          "sepal_width",
          "petal_length"
        ]
      },
      {
        "title": "Plot 4",
        "features": [
          "petal_length",
          "petal_width",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 5",
        "features": [
          "sepal_length",
          "petal_width"
        ]
      }
    ],
    "scale": true
  }
}
```
#### Command line
```python
agglomerative_clustering --config config_agglomerative_clustering.json --input_dataset_path dataset_agglomerative_clustering.csv --output_results_path ref_output_results_agglomerative_clustering.csv --output_plot_path ref_output_plot_agglomerative_clustering.png
```

## Neural_network_decode
Wrapper of the TensorFlow Keras LSTM method for decoding.
### Get help
Command:
```python
neural_network_decode -h
```
    usage: neural_network_decode [-h] [--config CONFIG] --input_decode_path INPUT_DECODE_PATH --input_model_path INPUT_MODEL_PATH --output_decode_path OUTPUT_DECODE_PATH [--output_predict_path OUTPUT_PREDICT_PATH]
    
    Wrapper of the TensorFlow Keras LSTM method for decoding.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_predict_path OUTPUT_PREDICT_PATH
                            Path to the output predict file. Accepted formats: csv.
    
    required arguments:
      --input_decode_path INPUT_DECODE_PATH
                            Path to the input decode dataset. Accepted formats: csv.
      --input_model_path INPUT_MODEL_PATH
                            Path to the input model. Accepted formats: h5.
      --output_decode_path OUTPUT_DECODE_PATH
                            Path to the output decode file. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_decode_path** (*string*): Path to the input decode dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/dataset_decoder.csv). Accepted formats: CSV
* **input_model_path** (*string*): Path to the input model. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/neural_networks/input_model_decoder.h5). Accepted formats: H5
* **output_decode_path** (*string*): Path to the output decode file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_decode_decoder.csv). Accepted formats: CSV
* **output_predict_path** (*string*): Path to the output predict file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/neural_networks/ref_output_predict_decoder.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_neural_network_decode.yml)
```python
properties:
  remove_tmp: false

```
#### Command line
```python
neural_network_decode --config config_neural_network_decode.yml --input_decode_path dataset_decoder.csv --input_model_path input_model_decoder.h5 --output_decode_path ref_output_decode_decoder.csv --output_predict_path ref_output_predict_decoder.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_neural_network_decode.json)
```python
{
  "properties": {
    "remove_tmp": false
  }
}
```
#### Command line
```python
neural_network_decode --config config_neural_network_decode.json --input_decode_path dataset_decoder.csv --input_model_path input_model_decoder.h5 --output_decode_path ref_output_decode_decoder.csv --output_predict_path ref_output_predict_decoder.csv
```

## Polynomial_regression
Wrapper of the scikit-learn LinearRegression method with PolynomialFeatures.
### Get help
Command:
```python
polynomial_regression -h
```
    usage: polynomial_regression [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn LinearRegression method with PolynomialFeatures.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Residual plot checks the error between actual values and predicted values. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/dataset_polynomial_regression.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_model_polynomial_regression.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_test_polynomial_regression.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Residual plot checks the error between actual values and predicted values. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_polynomial_regression.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **degree** (*integer*): (2) Polynomial degree..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_polynomial_regression.yml)
```python
properties:
  degree: 2
  independent_vars:
    columns:
    - LSTAT
    - RM
    - ZN
    - AGE
  scale: true
  target:
    column: MEDV
  test_size: 0.2

```
#### Command line
```python
polynomial_regression --config config_polynomial_regression.yml --input_dataset_path dataset_polynomial_regression.csv --output_model_path ref_output_model_polynomial_regression.pkl --output_test_table_path ref_output_test_polynomial_regression.csv --output_plot_path ref_output_plot_polynomial_regression.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_polynomial_regression.json)
```python
{
  "properties": {
    "independent_vars": {
      "columns": [
        "LSTAT",
        "RM",
        "ZN",
        "AGE"
      ]
    },
    "target": {
      "column": "MEDV"
    },
    "degree": 2,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
polynomial_regression --config config_polynomial_regression.json --input_dataset_path dataset_polynomial_regression.csv --output_model_path ref_output_model_polynomial_regression.pkl --output_test_table_path ref_output_test_polynomial_regression.csv --output_plot_path ref_output_plot_polynomial_regression.png
```

## Correlation_matrix
Generates a correlation matrix from a given dataset.
### Get help
Command:
```python
correlation_matrix -h
```
    usage: correlation_matrix [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_plot_path OUTPUT_PLOT_PATH
    
    Generates a correlation matrix from a given dataset
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the correlation matrix plot. Accepted formats: png.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_correlation_matrix.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the correlation matrix plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_plot_correlation_matrix.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Independent variables or columns from your dataset you want to compare. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_correlation_matrix.yml)
```python
properties:
  features:
    columns:
    - sepal_length
    - sepal_width
    - petal_length
    - petal_width

```
#### Command line
```python
correlation_matrix --config config_correlation_matrix.yml --input_dataset_path dataset_correlation_matrix.csv --output_plot_path ref_output_plot_correlation_matrix.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_correlation_matrix.json)
```python
{
  "properties": {
    "features": {
      "columns": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
      ]
    }
  }
}
```
#### Command line
```python
correlation_matrix --config config_correlation_matrix.json --input_dataset_path dataset_correlation_matrix.csv --output_plot_path ref_output_plot_correlation_matrix.png
```

## Dbscan
Wrapper of the scikit-learn DBSCAN method.
### Get help
Command:
```python
dbscan -h
```
    usage: dbscan [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn DBSCAN method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the clustering plot. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_results_path OUTPUT_RESULTS_PATH
                            Path to the clustered dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_dbscan.csv). Accepted formats: CSV
* **output_results_path** (*string*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_dbscan.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_dbscan.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **predictors** (*object*): ({}) Features or columns from your dataset you want to use for fitting. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **eps** (*number*): (0.5) The maximum distance between two samples for one to be considered as in the neighborhood of the other..
* **min_samples** (*integer*): (5) The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself..
* **metric** (*string*): (euclidean) The metric to use when calculating distance between instances in a feature array. .
* **plots** (*array*): (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ]..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_dbscan.yml)
```python
properties:
  eps: 1.4
  min_samples: 3
  plots:
  - features:
    - sepal_length
    - sepal_width
    title: Plot 1
  - features:
    - petal_length
    - petal_width
    title: Plot 2
  - features:
    - sepal_length
    - sepal_width
    - petal_length
    title: Plot 3
  - features:
    - petal_length
    - petal_width
    - sepal_width
    title: Plot 4
  - features:
    - sepal_length
    - petal_width
    title: Plot 5
  predictors:
    columns:
    - sepal_length
    - sepal_width
    - petal_length
    - petal_width
  scale: true

```
#### Command line
```python
dbscan --config config_dbscan.yml --input_dataset_path dataset_dbscan.csv --output_results_path ref_output_results_dbscan.csv --output_plot_path ref_output_plot_dbscan.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_dbscan.json)
```python
{
  "properties": {
    "predictors": {
      "columns": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
      ]
    },
    "eps": 1.4,
    "min_samples": 3,
    "plots": [
      {
        "title": "Plot 1",
        "features": [
          "sepal_length",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 2",
        "features": [
          "petal_length",
          "petal_width"
        ]
      },
      {
        "title": "Plot 3",
        "features": [
          "sepal_length",
          "sepal_width",
          "petal_length"
        ]
      },
      {
        "title": "Plot 4",
        "features": [
          "petal_length",
          "petal_width",
          "sepal_width"
        ]
      },
      {
        "title": "Plot 5",
        "features": [
          "sepal_length",
          "petal_width"
        ]
      }
    ],
    "scale": true
  }
}
```
#### Command line
```python
dbscan --config config_dbscan.json --input_dataset_path dataset_dbscan.csv --output_results_path ref_output_results_dbscan.csv --output_plot_path ref_output_plot_dbscan.png
```

## Resampling
Wrapper of the imblearn.combine methods.
### Get help
Command:
```python
resampling -h
```
    usage: resampling [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_dataset_path OUTPUT_DATASET_PATH
    
    Wrapper of the imblearn.combine methods.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_dataset_path OUTPUT_DATASET_PATH
                            Path to the output dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/resampling/dataset_resampling.csv). Accepted formats: CSV
* **output_dataset_path** (*string*): Path to the output dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/resampling/ref_output_resampling.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **method** (*string*): (None) Resampling method. It's a mandatory property. .
* **type** (*string*): (None) Type of oversampling. It's a mandatory property. .
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **evaluate** (*boolean*): (False) Whether or not to evaluate the dataset before and after applying the resampling..
* **evaluate_splits** (*integer*): (3) Number of folds to be applied by the Repeated Stratified K-Fold evaluation method. Must be at least 2..
* **evaluate_repeats** (*integer*): (3) Number of times Repeated Stratified K-Fold cross validator needs to be repeated..
* **n_bins** (*integer*): (5) Only for regression resampling. The number of classes that the user wants to generate with the target data..
* **balanced_binning** (*boolean*): (False) Only for regression resampling. Decides whether samples are to be distributed roughly equally across all classes..
* **sampling_strategy_over** (*object*): ({'target': 'auto'}) Sampling information applied in the dataset oversampling process. Formats: { "target": "auto" }, { "ratio": 0.3 } or { "dict": { 0: 300, 1: 200, 2: 100 } }. When "target", specify the class targeted by the resampling; the number of samples in the different classes will be equalized; possible choices are: minority (resample only the minority class), not minority (resample all classes but the minority class), not majority (resample all classes but the majority class), all (resample all classes), auto (equivalent to 'not majority'). When "ratio", it corresponds to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling (ONLY IN CASE OF BINARY CLASSIFICATION).  When "dict", the keys correspond to the targeted classes and the values correspond to the desired number of samples for each targeted class..
* **sampling_strategy_under** (*object*): ({'target': 'auto'}) Sampling information applied in the dataset cleaning process. Formats: { "target": "auto" } or { "list": [0, 2, 3] }. When "target", specify the class targeted by the resampling; the number of samples in the different classes will be equalized; possible choices are: majority (resample only the majority class), not minority (resample all classes but the minority class), not majority (resample all classes but the majority class), all (resample all classes), auto (equivalent to 'not minority'). When "list", the list contains the classes targeted by the resampling..
* **random_state_method** (*integer*): (5) Controls the randomization of the algorithm..
* **random_state_evaluate** (*integer*): (5) Controls the shuffling applied to the Repeated Stratified K-Fold evaluation method..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_resampling.yml)
```python
properties:
  evaluate: true
  method: smotenn
  n_bins: 10
  sampling_strategy_over:
    dict:
      4: 1000
      5: 1000
      6: 1000
      7: 1000
  sampling_strategy_under:
    list:
    - 0
    - 1
  target:
    column: VALUE
  type: regression

```
#### Command line
```python
resampling --config config_resampling.yml --input_dataset_path dataset_resampling.csv --output_dataset_path ref_output_resampling.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_resampling.json)
```python
{
  "properties": {
    "method": "smotenn",
    "type": "regression",
    "target": {
      "column": "VALUE"
    },
    "evaluate": true,
    "n_bins": 10,
    "sampling_strategy_over": {
      "dict": {
        "4": 1000,
        "5": 1000,
        "6": 1000,
        "7": 1000
      }
    },
    "sampling_strategy_under": {
      "list": [
        0,
        1
      ]
    }
  }
}
```
#### Command line
```python
resampling --config config_resampling.json --input_dataset_path dataset_resampling.csv --output_dataset_path ref_output_resampling.csv
```

## Scale_columns
Scales columns from a given dataset.
### Get help
Command:
```python
scale_columns -h
```
    usage: scale_columns [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_dataset_path OUTPUT_DATASET_PATH
    
    Scales columns from a given dataset
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_dataset_path OUTPUT_DATASET_PATH
                            Path to the output dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_scale.csv). Accepted formats: CSV
* **output_dataset_path** (*string*): Path to the output dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_scale.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **targets** (*object*): ({}) Independent variables or columns from your dataset you want to scale. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_scale_columns.yml)
```python
properties:
  targets:
    columns:
    - VALUE

```
#### Command line
```python
scale_columns --config config_scale_columns.yml --input_dataset_path dataset_scale.csv --output_dataset_path ref_output_scale.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_scale_columns.json)
```python
{
  "properties": {
    "targets": {
      "columns": [
        "VALUE"
      ]
    }
  }
}
```
#### Command line
```python
scale_columns --config config_scale_columns.json --input_dataset_path dataset_scale.csv --output_dataset_path ref_output_scale.csv
```

## Oversampling
Wrapper of most of the imblearn.over_sampling methods.
### Get help
Command:
```python
oversampling -h
```
    usage: oversampling [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_dataset_path OUTPUT_DATASET_PATH
    
    Wrapper of most of the imblearn.over_sampling methods.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_dataset_path OUTPUT_DATASET_PATH
                            Path to the output dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/resampling/dataset_resampling.csv). Accepted formats: CSV
* **output_dataset_path** (*string*): Path to the output dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/resampling/ref_output_oversampling.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **method** (*string*): (None) Oversampling method. It's a mandatory property. .
* **type** (*string*): (None) Type of oversampling. It's a mandatory property. .
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **evaluate** (*boolean*): (False) Whether or not to evaluate the dataset before and after applying the resampling..
* **evaluate_splits** (*integer*): (3) Number of folds to be applied by the Repeated Stratified K-Fold evaluation method. Must be at least 2..
* **evaluate_repeats** (*integer*): (3) Number of times Repeated Stratified K-Fold cross validator needs to be repeated..
* **n_bins** (*integer*): (5) Only for regression oversampling. The number of classes that the user wants to generate with the target data..
* **balanced_binning** (*boolean*): (False) Only for regression oversampling. Decides whether samples are to be distributed roughly equally across all classes..
* **sampling_strategy** (*object*): ({'target': 'auto'}) Sampling information to sample the data set. Formats: { "target": "auto" }, { "ratio": 0.3 }, { "dict": { 0: 300, 1: 200, 2: 100 } } or { "list": [0, 2, 3] }. When "target", specify the class targeted by the resampling; the number of samples in the different classes will be equalized; possible choices are: minority (resample only the minority class), not minority (resample all classes but the minority class), not majority (resample all classes but the majority class), all (resample all classes), auto (equivalent to 'not majority'). When "ratio", it corresponds to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling (ONLY IN CASE OF BINARY CLASSIFICATION).  When "dict", the keys correspond to the targeted classes, the values correspond to the desired number of samples for each targeted class. When "list", the list contains the classes targeted by the resampling..
* **k_neighbors** (*integer*): (5) Only for SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN. The number of nearest neighbours used to construct synthetic samples..
* **random_state_method** (*integer*): (5) Controls the randomization of the algorithm..
* **random_state_evaluate** (*integer*): (5) Controls the shuffling applied to the Repeated Stratified K-Fold evaluation method..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_oversampling.yml)
```python
properties:
  evaluate: true
  method: random
  n_bins: 10
  sampling_strategy:
    target: minority
  target:
    column: VALUE
  type: regression

```
#### Command line
```python
oversampling --config config_oversampling.yml --input_dataset_path dataset_resampling.csv --output_dataset_path ref_output_oversampling.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_oversampling.json)
```python
{
  "properties": {
    "method": "random",
    "type": "regression",
    "target": {
      "column": "VALUE"
    },
    "evaluate": true,
    "n_bins": 10,
    "sampling_strategy": {
      "target": "minority"
    }
  }
}
```
#### Command line
```python
oversampling --config config_oversampling.json --input_dataset_path dataset_resampling.csv --output_dataset_path ref_output_oversampling.csv
```

## Map_variables
Maps the values of a given dataset.
### Get help
Command:
```python
map_variables -h
```
    usage: map_variables [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_dataset_path OUTPUT_DATASET_PATH
    
    Maps the values of a given dataset.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_dataset_path OUTPUT_DATASET_PATH
                            Path to the output dataset. Accepted formats: csv.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_map_variables.csv). Accepted formats: CSV
* **output_dataset_path** (*string*): Path to the output dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_dataset_map_variables.csv). Accepted formats: CSV
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **targets** (*object*): ({}) Independent variables or columns from your dataset you want to drop. If None given, all the columns will be taken. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_map_variables.yml)
```python
properties:
  targets:
    columns:
    - target

```
#### Command line
```python
map_variables --config config_map_variables.yml --input_dataset_path dataset_map_variables.csv --output_dataset_path ref_output_dataset_map_variables.csv
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_map_variables.json)
```python
{
  "properties": {
    "targets": {
      "columns": [
        "target"
      ]
    }
  }
}
```
#### Command line
```python
map_variables --config config_map_variables.json --input_dataset_path dataset_map_variables.csv --output_dataset_path ref_output_dataset_map_variables.csv
```

## Dendrogram
Generates a dendrogram from a given dataset.
### Get help
Command:
```python
dendrogram -h
```
    usage: dendrogram [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_plot_path OUTPUT_PLOT_PATH
    
    Generates a dendrogram from a given dataset
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Path to the dendrogram plot. Accepted formats: png.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/utils/dataset_dendrogram.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Path to the dendrogram plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/utils/ref_output_plot_dendrogram.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **features** (*object*): ({}) Independent variables or columns from your dataset you want to compare. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_dendrogram.yml)
```python
properties:
  features:
    columns:
    - Satisfaction
    - Loyalty

```
#### Command line
```python
dendrogram --config config_dendrogram.yml --input_dataset_path dataset_dendrogram.csv --output_plot_path ref_output_plot_dendrogram.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_dendrogram.json)
```python
{
  "properties": {
    "features": {
      "columns": [
        "Satisfaction",
        "Loyalty"
      ]
    }
  }
}
```
#### Command line
```python
dendrogram --config config_dendrogram.json --input_dataset_path dataset_dendrogram.csv --output_plot_path ref_output_plot_dendrogram.png
```

## Random_forest_regressor
Wrapper of the scikit-learn RandomForestRegressor method.
### Get help
Command:
```python
random_forest_regressor -h
```
    usage: random_forest_regressor [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]
    
    Wrapper of the scikit-learn RandomForestRegressor method.
    
    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Configuration file
      --output_test_table_path OUTPUT_TEST_TABLE_PATH
                            Path to the test table file. Accepted formats: csv.
      --output_plot_path OUTPUT_PLOT_PATH
                            Residual plot checks the error between actual values and predicted values. Accepted formats: png.
    
    required arguments:
      --input_dataset_path INPUT_DATASET_PATH
                            Path to the input dataset. Accepted formats: csv.
      --output_model_path OUTPUT_MODEL_PATH
                            Path to the output model file. Accepted formats: pkl.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_dataset_path** (*string*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/dataset_random_forest_regressor.csv). Accepted formats: CSV
* **output_model_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_model_random_forest_regressor.pkl). Accepted formats: PKL
* **output_test_table_path** (*string*): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_test_random_forest_regressor.csv). Accepted formats: CSV
* **output_plot_path** (*string*): Residual plot checks the error between actual values and predicted values. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_random_forest_regressor.png). Accepted formats: PNG
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **independent_vars** (*object*): ({}) Independent variables you want to train from your dataset. You can specify either a list of columns names from your input dataset, a list of columns indexes or a range of columns indexes. Formats: { "columns": ["column1", "column2"] } or { "indexes": [0, 2, 3, 10, 11, 17] } or { "range": [[0, 20], [50, 102]] }. In case of mulitple formats, the first one will be picked..
* **target** (*object*): ({}) Dependent variable you want to predict from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **weight** (*object*): ({}) Weight variable from your dataset. You can specify either a column name or a column index. Formats: { "column": "column3" } or { "index": 21 }. In case of mulitple formats, the first one will be picked..
* **n_estimators** (*integer*): (10) The number of trees in the forest..
* **max_depth** (*integer*): (None) The maximum depth of the tree..
* **random_state_method** (*integer*): (5) Controls the randomness of the estimator..
* **random_state_train_test** (*integer*): (5) Controls the shuffling applied to the data before applying the split..
* **test_size** (*number*): (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0..
* **scale** (*boolean*): (False) Whether or not to scale the input dataset..
* **remove_tmp** (*boolean*): (True) Remove temporal files..
* **restart** (*boolean*): (False) Do not execute if output files exist..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_random_forest_regressor.yml)
```python
properties:
  independent_vars:
    range:
    - - 0
      - 5
    - - 7
      - 12
  max_depth: 5
  n_estimators: 10
  scale: true
  target:
    index: 13
  test_size: 0.2

```
#### Command line
```python
random_forest_regressor --config config_random_forest_regressor.yml --input_dataset_path dataset_random_forest_regressor.csv --output_model_path ref_output_model_random_forest_regressor.pkl --output_test_table_path ref_output_test_random_forest_regressor.csv --output_plot_path ref_output_plot_random_forest_regressor.png
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_ml/blob/master/biobb_ml/test/data/config/config_random_forest_regressor.json)
```python
{
  "properties": {
    "independent_vars": {
      "range": [
        [
          0,
          5
        ],
        [
          7,
          12
        ]
      ]
    },
    "target": {
      "index": 13
    },
    "n_estimators": 10,
    "max_depth": 5,
    "test_size": 0.2,
    "scale": true
  }
}
```
#### Command line
```python
random_forest_regressor --config config_random_forest_regressor.json --input_dataset_path dataset_random_forest_regressor.csv --output_model_path ref_output_model_random_forest_regressor.pkl --output_test_table_path ref_output_test_random_forest_regressor.csv --output_plot_path ref_output_plot_random_forest_regressor.png
```
