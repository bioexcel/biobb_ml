# BioBB Machine Learning Command Line Help

Generic usage:


```python
biobb_command [-h] [-c CONFIG] --in_file <in_file>  --out_file <out_file>
```

***

## linear_regression

Trains and tests a given dataset and saves the model and scaler for a linear regression.

### Get help


```python
linear_regression -h
```


```python
usage: linear_regression [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and saves the model and scaler for a linear regression.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/dataset_linear_regression.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_model_linear_regression.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_test_linear_regression.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Residual plot checks the error between actual values and predicted values. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_linear_regression.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['size', 'year', 'view']
  target: 'price'
  test_size: 0.2
```


```python
linear_regression --conf data/conf/linear_regression.yml --input_dataset_path data/input/dataset_linear_regression.csv --output_model_path data/output/output_model_linear_regression.pkl --output_test_table_path data/output/output_test_linear_regression.csv --output_plot_path data/output/output_plot_linear_regression.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["size", "year", "view"],
        "target": "price",
        "test_size": 0.2
    }
}
```


```python
linear_regression --conf data/conf/linear_regression.json --input_dataset_path data/input/dataset_linear_regression.csv --output_model_path data/output/output_model_linear_regression.pkl --output_test_table_path data/output/output_test_linear_regression.csv --output_plot_path data/output/output_plot_linear_regression.png
```

## polynomial_regression

Trains and tests a given dataset and saves the model and scaler for a polynomial regression.

### Get help


```python
polynomial_regression -h
```


```python
usage: polynomial_regression [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and saves the model and scaler for a polynomial regression.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/dataset_polynomial_regression.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_model_polynomial_regression.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_test_polynomial_regression.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Residual plot checks the error between actual values and predicted values. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_polynomial_regression.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **degree** (*int*) - (2) Polynomial degree.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['LSTAT','RM']
  target: 'MEDV'
  degree: 2
  test_size: 0.2
```


```python
polynomial_regression --conf data/conf/polynomial_regression.yml --input_dataset_path data/input/dataset_polynomial_regression.csv --output_model_path data/output/output_model_polynomial_regression.pkl --output_test_table_path data/output/output_test_polynomial_regression.csv --output_plot_path data/output/output_plot_polynomial_regression.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["LSTAT", "RM"],
        "target": "MEDV",
        "degree": 2,
        "test_size": 0.2
    }
}
```


```python
polynomial_regression --conf data/conf/polynomial_regression.json --input_dataset_path data/input/dataset_polynomial_regression.csv --output_model_path data/output/output_model_polynomial_regression.pkl --output_test_table_path data/output/output_test_polynomial_regression.csv --output_plot_path data/output/output_plot_polynomial_regression.png
```

## random_forest_regressor

Trains and tests a given dataset and saves the model and scaler for a random forest regressor.

### Get help


```python
random_forest_regressor -h
```


```python
usage: random_forest_regressor [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and saves the model and scaler for a random forest regressor.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/dataset_random_forest_regressor.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_model_random_forest_regressor.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_test_random_forest_regressor.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Residual plot checks the error between actual values and predicted values. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_random_forest_regressor.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **n_estimators** (*int*) - (10) The number of trees in the forest.
* **max_depth** (*int*) - (5) The maximum depth of the tree.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['LSTAT','RM']
  target: 'MEDV'
  n_estimators: 10
  max_depth: 5
  test_size: 0.2
```


```python
random_forest_regressor --conf data/conf/random_forest_regressor.yml --input_dataset_path data/input/dataset_random_forest_regressor.csv --output_model_path data/output/output_model_random_forest_regressor.pkl --output_test_table_path data/output/output_test_random_forest_regressor.csv --output_plot_path data/output/output_plot_random_forest_regressor.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["LSTAT", "RM"],
        "target": "MEDV",
        "n_estimators": 10,
        "max_depth": 5,
        "test_size": 0.2
    }
}
```


```python
random_forest_regressor --conf data/conf/random_forest_regressor.json --input_dataset_path data/input/dataset_random_forest_regressor.csv --output_model_path data/output/output_model_random_forest_regressor.pkl --output_test_table_path data/output/output_test_random_forest_regressor.csv --output_plot_path data/output/output_plot_random_forest_regressor.png
```

## regression_predict

Makes predictions from a given model.

### Get help


```python
regression_predict -h
```


```python
usage: regression_predict [-h] [--config CONFIG] --input_model_path INPUT_MODEL_PATH --output_results_path OUTPUT_RESULTS_PATH

Makes predictions from a given model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file

required arguments:
  --input_model_path INPUT_MODEL_PATH
                        Path to the input model. Accepted formats: pkl.
  --output_results_path OUTPUT_RESULTS_PATH
                        Path to the output results file. Accepted formats: csv.
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_model_path** (*str*): Path to the input model. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/regression/model_regression_predict.pkl). Accepted formats: pkl.
* **output_results_path** (*str*) (Optional): Path to the output results file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_regression_predict.csv). Accepted formats: csv.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictions: [ 
    { 'LSTAT': 4.98, 'RM': 6.575 }, 
    { 'LSTAT': 9.14, 'RM': 6.421 }
  ]
```


```python
regression_predict --conf data/conf/regression_predict.yml --input_model_path data/input/model_regression_predict.pkl --output_results_path data/output/output_regression_predict.csv
```

### JSON

#### File config


```python
{
    "properties": {
        "predictions": [ 
            { "LSTAT": 4.98, "RM": 6.575 }, 
            { "LSTAT": 9.14, "RM": 6.421 }
        ]
    }
}
```


```python
regression_predict --conf data/conf/regression_predict.json --input_model_path data/input/model_regression_predict.pkl --output_results_path data/output/output_regression_predict.csv
```

## decision_tree

Trains and tests a given dataset and saves the model and scaler for a decision tree classification.

### Get help


```python
decision_tree -h
```


```python
usage: decision_tree [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and saves the model and scaler for a decision tree classification.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_decision_tree.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_decision_tree.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_decision_tree.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_decision_tree.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **criterion** (*string*) - ("gini") The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. Values: gini, entropy.
* **max_depth** (*int*) - (4) The maximum depth of the model. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
* **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['duration', 'interest_rate', 'march', 'credit', 'previous']
  target: 'y'
  criterion: 'entropy'
  max_depth: 4
  normalize_cm: false
  test_size: 0.2
```


```python
decision_tree --conf data/conf/decision_tree.yml --input_dataset_path data/input/dataset_decision_tree.csv --output_model_path data/output/output_model_decision_tree.pkl --output_test_table_path data/output/output_test_decision_tree.csv --output_plot_path data/output/output_plot_decision_tree.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["duration", "interest_rate", "march", "credit", "previous"],
        "target": "y",
        "criterion": "entropy",
        "max_depth": 4,
        "normalize_cm": false,
        "test_size": 0.2
    }
}
```


```python
decision_tree --conf data/conf/decision_tree.json --input_dataset_path data/input/dataset_decision_tree.csv --output_model_path data/output/output_model_decision_tree.pkl --output_test_table_path data/output/output_test_decision_tree.csv --output_plot_path data/output/output_plot_decision_tree.png
```

## k_neighbors_coefficient

Trains and tests a given dataset and calculates best K coefficient for a k-nearest neighbors classification.

### Get help


```python
k_neighbors_coefficient -h
```


```python
usage: k_neighbors_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and calculates best K coefficient for a k-nearest neighbors classification.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_k_neighbors_coefficient.csv). Accepted formats: csv.
* **output_results_path** (*str*) (Optional): Path to the accuracy values list. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_k_neighbors_coefficient.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the accuracy plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_k_neighbors_coefficient.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **metric** (*string*) - ("minkowski") The distance metric to use for the tree. Values: euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobi.
* **max_neighbors** (*int*) - (6) Maximum number of neighbors to use by default for kneighbors queries.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']
  target: 'custcat'
  metric: 'minkowski'
  max_neighbors: 15
  test_size: 0.2
```


```python
k_neighbors_coefficient --conf data/conf/k_neighbors_coefficients.yml --input_dataset_path data/input/dataset_k_neighbors_coefficient.csv --output_results_path data/output/output_test_k_neighbors_coefficient.csv --output_plot_path data/output/output_plot_k_neighbors_coefficient.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["region", "tenure", "age", "marital", "address", "income", "ed", "employ", "retire", "gender", "reside"],
        "target": "custcat",
        "metric": "minkowski",
        "max_neighbors": 15,
        "test_size": 0.2
    }
}
```


```python
k_neighbors_coefficient --conf data/conf/k_neighbors_coefficients.json --input_dataset_path data/input/dataset_k_neighbors_coefficient.csv --output_results_path data/output/output_test_k_neighbors_coefficient.csv --output_plot_path data/output/output_plot_k_neighbors_coefficient.png
```

## k_neighbors

Trains and tests a given dataset and saves the model and scaler for a k-nearest neighbors classification.

### Get help


```python
k_neighbors -h
```


```python
usage: k_neighbors [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and saves the model and scaler for a k-nearest neighbors classification.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_k_neighbors.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_k_neighbors.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_k_neighbors.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_k_neighbors.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **metric** (*string*) - ("minkowski") The distance metric to use for the tree. Values: euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobi.
* **n_neighbors** (*int*) - (6) Number of neighbors to use by default for kneighbors queries.
* **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['duration', 'interest_rate', 'march', 'credit', 'previous']
  target: 'y'
  metric: 'minkowski'
  n_neighbors: 5
  normalize_cm: false
  test_size: 0.2
```


```python
k_neighbors --conf data/conf/kneighbors.yml --input_dataset_path data/input/dataset_k_neighbors.csv --output_model_path data/output/output_model_k_neighbors.pkl --output_test_table_path data/output/output_test_k_neighbors.csv --output_plot_path data/output/output_plot_k_neighbors.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["duration", "interest_rate", "march", "credit", "previous"],
        "target": "y",
        "metric": "minkowski",
        "n_neighbors": 5,
        "normalize_cm": false,
        "test_size": 0.2
    }
}
```


```python
k_neighbors --conf data/conf/kneighbors.json --input_dataset_path data/input/dataset_k_neighbors.csv --output_model_path data/output/output_model_k_neighbors.pkl --output_test_table_path data/output/output_test_k_neighbors.csv --output_plot_path data/output/output_plot_k_neighbors.png
```

## logistic_regression

Trains and tests a given dataset and saves the model and scaler for a logistic regression.

### Get help


```python
logistic_regression -h
```


```python
usage: logistic_regression [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and saves the model and scaler for a logistic regression.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_logistic_regression.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_logistic_regression.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_logistic_regression.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_logistic_regression.csv). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **solver** (*string*) - ("liblinear") Numerical optimizer to find parameters. Values: newton-cg, lbfgs, liblinear, sag, saga
* **c_parameter** (*float*) - (0.01) Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
* **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['mean area', 'mean compactness']
  target: 'benign'
  solver: 'liblinear'
  c_parameter: 0.01
  normalize_cm: false
  test_size: 0.2
```


```python
logistic_regression --conf data/conf/logistic_regression.yml --input_dataset_path data/input/dataset_logistic_regression.csv --output_model_path data/output/output_model_logistic_regression.pkl --output_test_table_path data/output/output_test_logistic_regression.csv --output_plot_path data/output/output_plot_logistic_regression.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["mean area", "mean compactness"],
        "target": "benign",
        "solver": "liblinear",
        "c_parameter": 0.01,
        "normalize_cm": false,
        "test_size": 0.2
    }
}
```


```python
logistic_regression --conf data/conf/logistic_regression.json --input_dataset_path data/input/dataset_logistic_regression.csv --output_model_path data/output/output_model_logistic_regression.pkl --output_test_table_path data/output/output_test_logistic_regression.csv --output_plot_path data/output/output_plot_logistic_regression.png
```

## random_forest_classifier

Trains and tests a given dataset and saves the model and scaler for a random forest classifier.

### Get help


```python
random_forest_classifier -h
```


```python
usage: random_forest_classifier [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and calculates coefficients and predictions for a random forest classifier.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_random_forest_classifier.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_random_forest_classifier.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_random_forest_classifier.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_random_forest_classifier.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **n_estimators** (*int*) - (100) The number of trees in the forest.
* **bootstrap** (*bool*) - (True) Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
* **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
  target: 'Drug'
  n_estimators: 100
  bootstrap: true
  normalize_cm: false
  test_size: 0.2
```


```python
random_forest_classifier --conf data/conf/random_forest_classifier.yml --input_dataset_path data/input/dataset_random_forest_classifier.csv --output_model_path data/output/output_model_random_forest_classifier.pkl --output_test_table_path data/output/output_test_random_forest_classifier.csv --output_plot_path data/output/output_plot_random_forest_classifier.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"],
        "target": "Drug",
        "n_estimators": 100,
        "bootstrap": true,
        "normalize_cm": false,
        "test_size": 0.2
    }
}
```


```python
random_forest_classifier --conf data/conf/random_forest_classifier.json --input_dataset_path data/input/dataset_random_forest_classifier.csv --output_model_path data/output/output_model_random_forest_classifier.pkl --output_test_table_path data/output/output_test_random_forest_classifier.csv --output_plot_path data/output/output_plot_random_forest_classifier.png
```

## support_vector_machine

Trains and tests a given dataset and saves the model and scaler for a support vector machine.

### Get help


```python
support_vector_machine -h
```


```python
usage: support_vector_machine [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_model_path OUTPUT_MODEL_PATH [--output_test_table_path OUTPUT_TEST_TABLE_PATH] [--output_plot_path OUTPUT_PLOT_PATH]

Trains and tests a given dataset and saves the model and scaler for a support vector machine.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/dataset_support_vector_machine.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_model_support_vector_machine.pkl). Accepted formats: pkl.
* **output_test_table_path** (*str*) (Optional): Path to the test table file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_test_support_vector_machine.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the statistics plot. If target is binary it shows confusion matrix, distributions of the predicted probabilities of both classes and ROC curve. If target is non-binary it shows confusion matrix. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_plot_support_vector_machine.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **independent_vars** (*list*) - (None) Independent variables or columns from your dataset you want to train.
* **target** (*string*) - (None) Dependent variable or column from your dataset you want to predict.
* **kernel** (*string*) - ("rbf") Specifies the kernel type to be used in the algorithm. Values: linear, poly, rbf, sigmoid, precomputed.
* **normalize_cm** (*bool*) - (False) Whether or not to normalize the confusion matrix.
* **test_size** (*float*) - (0.2) Represents the proportion of the dataset to include in the test split. It should be between 0.0 and 1.0.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  independent_vars: ['duration', 'interest_rate', 'march', 'credit', 'previous']
  target: 'y'
  kernel: 'rbf'
  normalize_cm: false                              
  test_size: 0.2
```


```python
support_vector_machine --conf data/conf/support_vector_machine.yml --input_dataset_path data/input/dataset_support_vector_machine.csv --output_model_path data/output/output_model_support_vector_machine.pkl --output_test_table_path data/output/output_test_support_vector_machine.csv --output_plot_path data/output/output_plot_support_vector_machine.png
```

### JSON

#### File config


```python
{
    "properties": {
        "independent_vars": ["duration", "interest_rate", "march", "credit", "previous"],
        "target": "y",
        "kernel": "rbf",
        "normalize_cm": false,
        "test_size": 0.2
    }
}
```


```python
support_vector_machine --conf data/conf/support_vector_machine.json --input_dataset_path data/input/dataset_support_vector_machine.csv --output_model_path data/output/output_model_support_vector_machine.pkl --output_test_table_path data/output/output_test_support_vector_machine.csv --output_plot_path data/output/output_plot_support_vector_machine.png
```

## classification_predict

Makes predictions from a given model.

### Get help


```python
classification_predict -h
```


```python
usage: classification_predict [-h] [--config CONFIG] --input_model_path INPUT_MODEL_PATH --output_results_path OUTPUT_RESULTS_PATH

Makes predictions from a given model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file

required arguments:
  --input_model_path INPUT_MODEL_PATH
                        Path to the input model. Accepted formats: pkl.
  --output_results_path OUTPUT_RESULTS_PATH
                        Path to the output results file. Accepted formats: csv.
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_model_path** (*str*): Path to the input model. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/classification/model_classification_predict.pkl). Accepted formats: pkl.
* **output_results_path** (*str*) (Optional): Path to the output results file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/classification/ref_output_classification_predict.csv). Accepted formats: csv.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:                           
  predictions: [ 
    { 'duration': 117.000, 'interest_rate': 1.334, 'march': 1.000, 'credit': 0.000, 'previous': 0.000 },
    { 'duration': 274.000, 'interest_rate': 0.767, 'march': 0.000, 'credit': 0.000, 'previous': 1.000 }, 
    { 'duration': 167.000, 'interest_rate': 4.858, 'march': 1.000, 'credit': 0.000, 'previous': 0.000 }, 
  ]
```


```python
classification_predict --conf data/conf/classification_predict.yml --input_model_path data/input/model_classification_predict.pkl --output_results_path data/output/output_classification_predict.csv
```

### JSON

#### File config


```python
{
    "properties": {
        "predictions": [ 
            { "duration": 117.000, "interest_rate": 1.334, "march": 1.000, "credit": 0.000, "previous": 0.000 },
            { "duration": 274.000, "interest_rate": 0.767, "march": 0.000, "credit": 0.000, "previous": 1.000 },
            { "duration": 167.000, "interest_rate": 4.858, "march": 1.000, "credit": 0.000, "previous": 0.000 }
        ]
    }
}
```


```python
classification_predict --conf data/conf/classification_predict.json --input_model_path data/input/model_classification_predict.pkl --output_results_path data/output/output_classification_predict.csv
```

## agglomerative_coefficient

Clusters a given dataset and calculates best K coefficient for an agglomerative clustering.

### Get help


```python
agglomerative_coefficient -h
```


```python
usage: agglomerative_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Clusters a given dataset and calculates best K coefficient for a k-means clustering.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_agglomerative_coefficient.csv). Accepted formats: csv.
* **output_results_path** (*str*) (Optional):  Path to the gap values list.  File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_agglomerative_coefficient.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the elbow method and gap statistics plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_agglomerative_coefficient.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
* **max_clusters** (*int*) - (6) Maximum number of clusters to use by default for kmeans queries.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictors: ['sepal_length', 'sepal_width']
  max_clusters: 10
```


```python
agglomerative_coefficient --conf data/conf/agglomerative_coefficient.yml --input_dataset_path data/input/dataset_agglomerative_coefficient.csv --output_results_path data/output/output_results_agglomerative_coefficient.csv --output_plot_path data/output/output_plot_agglomerative_coefficient.png
```

### JSON

#### File config


```python
{
    "properties": {
        "predictors": ["sepal_length", "sepal_width"],
        "max_clusters": 10
    }
}
```


```python
agglomerative_coefficient --conf data/conf/agglomerative_coefficient.json --input_dataset_path data/input/dataset_agglomerative_coefficient.csv --output_results_path data/output/output_results_agglomerative_coefficient.csv --output_plot_path data/output/output_plot_agglomerative_coefficient.png
```

## agglomerative_clustering

Clusters a given dataset with agglomerative clustering method.

### Get help


```python
agglomerative_clustering -h
```


```python
usage: agglomerative_clustering [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Clusters a given dataset with k-means clustering method.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_agglomerative_clustering.csv). Accepted formats: csv.
* **output_results_path** (*str*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_agglomerative_clustering.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_agglomerative_clustering.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
* **clusters** (*int*) - (3) The number of clusters to form as well as the number of centroids to generate.
* **linkage** (*int*) - ("ward") The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. Values: ward, complete, average, single.
* **plots** (*list*) - (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ].
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictors: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
  clusters: 3
  linkage: average
  plots: [
    { 
      'title': 'Plot 1',
      'features': ['sepal_length', 'sepal_width']
    },
    { 
      'title': 'Plot 2',
      'features': ['petal_length', 'petal_width']
    },
    { 
      'title': 'Plot 3',
      'features': ['sepal_length', 'sepal_width', 'petal_length']
    },
    { 
      'title': 'Plot 4',
      'features': ['petal_length', 'petal_width', 'sepal_width']
    },
    { 
      'title': 'Plot 5',
      'features': ['sepal_length', 'petal_width']
    }
  ]
```


```python
agglomerative_clustering --conf data/conf/agglomerative_clustering.yml --input_dataset_path data/input/dataset_agglomerative_clustering.csv --output_results_path data/output/output_results_agglomerative_clustering.csv --output_plot_path data/output/output_plot_agglomerative_clustering.png
```

### JSON

#### File config


```python
{
    "properties": {
        "predictors": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "clusters": 3,
        "linkage": "average",
        "plots": [
            { 
              "title": "Plot 1",
              "features": ["sepal_length", "sepal_width"]
            },
            { 
              "title": "Plot 2",
              "features": ["petal_length", "petal_width"]
            },
            { 
              "title": "Plot 3",
              "features": ["sepal_length", "sepal_width", "petal_length"]
            },
            { 
              "title": "Plot 4",
              "features": ["petal_length", "petal_width", "sepal_width"]
            },
            { 
              "title": "Plot 5",
              "features": ["sepal_length", "petal_width"]
            }
        ]
    }
}
```


```python
agglomerative_clustering --conf data/conf/agglomerative_clustering.json --input_dataset_path data/input/dataset_agglomerative_clustering.csv --output_results_path data/output/output_results_agglomerative_clustering.csv --output_plot_path data/output/output_plot_agglomerative_clustering.png
```

## dbscan

Clusters a given dataset with DBSCAN clustering method.

### Get help


```python
dbscan -h
```


```python
usage: dbscan [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Clusters a given dataset with DBSCAN clustering method.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_dbscan.csv). Accepted formats: csv.
* **output_results_path** (*str*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_dbscan.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_dbscan.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
* **eps** (*float*) - (0.5) The maximum distance between two samples for one to be considered as in the neighborhood of the other.
* **min_samples** (*int*) - (5) The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
* **plots** (*list*) - (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ].
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictors: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
  eps: 1.4
  min_samples: 3
  plots: [
    { 
      'title': 'Plot 1',
      'features': ['sepal_length', 'sepal_width']
    },
    { 
      'title': 'Plot 2',
      'features': ['petal_length', 'petal_width']
    },
    { 
      'title': 'Plot 3',
      'features': ['sepal_length', 'sepal_width', 'petal_length']
    },
    { 
      'title': 'Plot 4',
      'features': ['petal_length', 'petal_width', 'sepal_width']
    },
    { 
      'title': 'Plot 5',
      'features': ['sepal_length', 'petal_width']
    }
  ]
```


```python
dbscan --conf data/conf/dbscan.yml --input_dataset_path data/input/dataset_dbscan.csv --output_results_path data/output/output_results_dbscan.csv --output_plot_path data/output/output_plot_dbscan.png
```

### JSON

#### File config


```python
{
    "properties": {
        "predictors": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "eps": 1.4,
        "min_samples":3,
        "plots": [
            { 
              "title": "Plot 1",
              "features": ["sepal_length", "sepal_width"]
            },
            { 
              "title": "Plot 2",
              "features": ["petal_length", "petal_width"]
            },
            { 
              "title": "Plot 3",
              "features": ["sepal_length", "sepal_width", "petal_length"]
            },
            { 
              "title": "Plot 4",
              "features": ["petal_length", "petal_width", "sepal_width"]
            },
            { 
              "title": "Plot 5",
              "features": ["sepal_length", "petal_width"]
            }
        ]
    }
}
```


```python
dbscan --conf data/conf/dbscan.json --input_dataset_path data/input/dataset_dbscan.csv --output_results_path data/output/output_results_dbscan.csv --output_plot_path data/output/output_plot_dbscan.png
```

## k_means_coefficient

Clusters a given dataset and calculates best K coefficient for a k-means clustering.

### Get help


```python
k_means_coefficient -h
```


```python
usage: k_means_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Clusters a given dataset and calculates best K coefficient for a k-means clustering.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_k_means_coefficient.csv). Accepted formats: csv.
* **output_results_path** (*str*) (Optional):  Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_k_means_coefficient.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the elbow method and gap statistics plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_k_means_coefficient.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
* **max_clusters** (*int*) - (6) Maximum number of clusters to use by default for kmeans queries.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictors: ['sepal_length', 'sepal_width']
  max_clusters: 10
```


```python
k_means_coefficient --conf data/conf/k_means_coefficient.yml --input_dataset_path data/input/dataset_k_means_coefficient.csv --output_results_path data/output/output_results_k_means_coefficient.csv --output_plot_path data/output/output_plot_k_means_coefficient.png
```

### JSON

#### File config


```python
{
    "properties": {
        "predictors": ["sepal_length", "sepal_width"],
        "max_clusters": 10
    }
}
```


```python
k_means_coefficient --conf data/conf/k_means_coefficient.json --input_dataset_path data/input/dataset_k_means_coefficient.csv --output_results_path data/output/output_results_k_means_coefficient.csv --output_plot_path data/output/output_plot_k_means_coefficient.png
```

## k_means

Clusters a given dataset and saves a model with k-means clustering method.

### Get help


```python
k_means -h
```


```python
usage: k_means [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH --output_model_path OUTPUT_MODEL_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Clusters a given dataset and saves a model with k-means clustering method.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_k_means.csv). Accepted formats: csv.
* **output_results_path** (*str*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_k_means.csv). Accepted formats: csv.
* **output_model_path** (*str*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_model_k_means.pkl). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_k_means.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
* **clusters** (*int*) - (3) The number of clusters to form as well as the number of centroids to generate.
* **plots** (*list*) - (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ].
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictors: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
  clusters: 3
  plots: [
    { 
      'title': 'Plot 1',
      'features': ['sepal_length', 'sepal_width']
    },
    { 
      'title': 'Plot 2',
      'features': ['petal_length', 'petal_width']
    },
    { 
      'title': 'Plot 3',
      'features': ['sepal_length', 'sepal_width', 'petal_length']
    },
    { 
      'title': 'Plot 4',
      'features': ['petal_length', 'petal_width', 'sepal_width']
    },
    { 
      'title': 'Plot 5',
      'features': ['sepal_length', 'petal_width']
    }
  ]
```


```python
k_means --conf data/conf/kmeans.yml --input_dataset_path data/input/dataset_k_means.csv --output_results_path data/output/output_results_k_means.csv --output_model_path data/output/output_model_k_means.pkl --output_plot_path data/output/output_plot_k_means.png
```

### JSON

#### File config


```python
{
    "properties": {
        "predictors": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "clusters": 3,
        "plots": [
            { 
              "title": "Plot 1",
              "features": ["sepal_length", "sepal_width"]
            },
            { 
              "title": "Plot 2",
              "features": ["petal_length", "petal_width"]
            },
            { 
              "title": "Plot 3",
              "features": ["sepal_length", "sepal_width", "petal_length"]
            },
            { 
              "title": "Plot 4",
              "features": ["petal_length", "petal_width", "sepal_width"]
            },
            { 
              "title": "Plot 5",
              "features": ["sepal_length", "petal_width"]
            }
        ]
    }
}
```


```python
k_means --conf data/conf/kmeans.json --input_dataset_path data/input/dataset_k_means.csv --output_results_path data/output/output_results_k_means.csv --output_model_path data/output/output_model_k_means.pkl --output_plot_path data/output/output_plot_k_means.png
```

## spectral_coefficient

Clusters a given dataset and calculates best K coefficient for a spectral clustering.

### Get help


```python
spectral_coefficient -h
```


```python
usage: spectral_coefficient [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Clusters a given dataset and calculates best K coefficient for a k-means clustering.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_spectral_coefficient.csv). Accepted formats: csv.
* **output_results_path** (*str*) (Optional):  Table with WCSS (elbow method), Gap and Silhouette coefficients for each cluster. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_spectral_coefficient.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the elbow method and gap statistics plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_spectral_coefficient.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
* **max_clusters** (*int*) - (6) Maximum number of clusters to use by default for kmeans queries.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictors: ['sepal_length', 'sepal_width']
  max_clusters: 10
```


```python
spectral_coefficient --conf data/conf/spectral_coefficient.yml --input_dataset_path data/input/dataset_spectral_coefficient.csv --output_results_path data/output/output_results_spectral_coefficient.csv --output_plot_path data/output/output_plot_spectral_coefficient.png
```

### JSON

#### File config


```python
{
    "properties": {
        "predictors": ["sepal_length", "sepal_width"],
        "max_clusters": 10
    }
}
```


```python
spectral_coefficient --conf data/conf/spectral_coefficient.json --input_dataset_path data/input/dataset_spectral_coefficient.csv --output_results_path data/output/output_results_spectral_coefficient.csv --output_plot_path data/output/output_plot_spectral_coefficient.png
```

## spectral_clustering

Clusters a given dataset with spectral clustering method.

### Get help


```python
spectral_clustering -h
```


```python
usage: spectral_clustering [-h] [--config CONFIG] --input_dataset_path INPUT_DATASET_PATH --output_results_path OUTPUT_RESULTS_PATH [--output_plot_path OUTPUT_PLOT_PATH]

Clusters a given dataset with spectral clustering method.

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
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_dataset_path** (*str*): Path to the input dataset. File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/dataset_spectral_clustering.csv). Accepted formats: csv.
* **output_results_path** (*str*): Path to the clustered dataset. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_spectral_clustering.csv). Accepted formats: csv.
* **output_plot_path** (*str*) (Optional): Path to the clustering plot. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_plot_spectral_clustering.png). Accepted formats: png.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictors** (*list*) - (None) Features or columns from your dataset you want to use for fitting.
* **clusters** (*int*) - (3) The number of clusters to form as well as the number of centroids to generate.
* **affinity** (*string*) - ("rbf") How to construct the affinity matrix. Values:  nearest_neighbors, rbf, precomputed, precomputed_nearest_neighbors.
* **plots** (*list*) - (None) List of dictionaries with all plots you want to generate. Only 2D or 3D plots accepted. Format: [ { 'title': 'Plot 1', 'features': ['feat1', 'feat2'] } ].
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictors: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
  clusters: 3
  affinity: 'nearest_neighbors'
  plots: [
    { 
      'title': 'Plot 1',
      'features': ['sepal_length', 'sepal_width']
    },
    { 
      'title': 'Plot 2',
      'features': ['petal_length', 'petal_width']
    },
    { 
      'title': 'Plot 3',
      'features': ['sepal_length', 'sepal_width', 'petal_length']
    },
    { 
      'title': 'Plot 4',
      'features': ['petal_length', 'petal_width', 'sepal_width']
    },
    { 
      'title': 'Plot 5',
      'features': ['sepal_length', 'petal_width']
    }
  ]
```


```python
spectral_clustering --conf data/conf/spectral_clustering.yml --input_dataset_path data/input/dataset_spectral_clustering.csv --output_results_path data/output/output_results_spectral_clustering.csv --output_plot_path data/output/output_plot_spectral_clustering.png
```

### JSON

#### File config


```python
{
    "properties": {
        "predictors": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "clusters": 3,
        "affinity": "nearest_neighbors",
        "plots": [
            { 
              "title": "Plot 1",
              "features": ["sepal_length", "sepal_width"]
            },
            { 
              "title": "Plot 2",
              "features": ["petal_length", "petal_width"]
            },
            { 
              "title": "Plot 3",
              "features": ["sepal_length", "sepal_width", "petal_length"]
            },
            { 
              "title": "Plot 4",
              "features": ["petal_length", "petal_width", "sepal_width"]
            },
            { 
              "title": "Plot 5",
              "features": ["sepal_length", "petal_width"]
            }
        ]
    }
}
```


```python
spectral_clustering --conf data/conf/spectral_clustering.json --input_dataset_path data/input/dataset_spectral_clustering.csv --output_results_path data/output/output_results_spectral_clustering.csv --output_plot_path data/output/output_plot_spectral_clustering.png
```

## clustering_predict

Makes predictions from a given model.

### Get help


```python
clustering_predict -h
```


```python
usage: clustering_predict [-h] [--config CONFIG] --input_model_path INPUT_MODEL_PATH --output_results_path OUTPUT_RESULTS_PATH

Makes predictions from a given model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file

required arguments:
  --input_model_path INPUT_MODEL_PATH
                        Path to the input model. Accepted formats: pkl.
  --output_results_path OUTPUT_RESULTS_PATH
                        Path to the output results file. Accepted formats: csv.
```

### I / O Arguments

Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:

* **input_model_path** (*str*): Path to the input model.File type: input. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/data/clustering/model_clustering_predict.pkl). Accepted formats: pkl.
* **output_results_path** (*str*): Path to the output results file. File type: output. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/clustering/ref_output_results_clustering_predict.csv). Accepted formats: csv.

### Config

Syntax: input_parameter (datatype) - (default_value) Definition


Config parameters for this building block:

* **predictions** (*list*) - (None) List of dictionaries with all values you want to predict targets.
* **remove_tmp** (*bool*) - (True) [WF property] Remove temporal files.
* **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

### YAML

#### File config


```python
properties:
  predictions: [
    { 'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2 },
    { 'sepal_length': 6.7, 'sepal_width': 3.0, 'petal_length': 5.2, 'petal_width': 2.3 },
    { 'sepal_length': 6.3, 'sepal_width': 2.5, 'petal_length': 5.0, 'petal_width': 1.9 }
  ]
```


```python
clustering_predict --conf data/conf/clustering_predict.yml --input_model_path data/input/model_clustering_predict.pkl --output_results_path data/output/output_results_clustering_predict.csv
```

### JSON

#### File config


```python
{
    "properties": {
        "predictions": [
            { "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 },
            { "sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3 },
            { "sepal_length": 6.3, "sepal_width": 2.5, "petal_length": 5.0, "petal_width": 1.9 }
        ]
    }
}
```


```python
clustering_predict --conf data/conf/clustering_predict.json --input_model_path data/input/model_clustering_predict.pkl --output_results_path data/output/output_results_clustering_predict.csv
```
