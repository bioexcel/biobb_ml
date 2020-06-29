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
* **output_plot_path** (*str*) (Optional): Residual plot checks the error between actual values and predicted values. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_linear_regression.png). File type: output. Accepted formats: png.

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
* **output_plot_path** (*str*) (Optional): Residual plot checks the error between actual values and predicted values. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_polynomial_regression.png). File type: output. Accepted formats: png.

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
* **output_plot_path** (*str*) (Optional): Residual plot checks the error between actual values and predicted values. [Sample file](https://github.com/bioexcel/biobb_ml/raw/master/biobb_ml/test/reference/regression/ref_output_plot_random_forest_regressor.png). File type: output. Accepted formats: png.

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


```python

```
