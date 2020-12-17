# Biobb Machine Learning changelog

## What's new in version [3.0.3](https://github.com/bioexcel/biobb_ml/releases/tag/v3.0.3)?
In version 3.0.3 the dependency biobb_common has been updated to 3.5.1 version. Also, there has been implemented the new version of docstrings, therefore the JSON Schemas have been modified. There has been implemented some new utils as well the new Resampling package. Finally, the method for inserting new datasets to the tools has been improved

### New features

* Update to biobb_common 3.5.1
* New extended and improved JSON schemas (Galaxy and CWL-compliant)
* New Drop Columns and Scale Columns utils
* New Resampling module: Oversampling, Undersampling and Resampling.

### Other changes

* New docstrings
* New input datasets method: now unlabeled datasets are supported.

## What's new in version [3.0.2](https://github.com/bioexcel/biobb_ml/releases/tag/v3.0.2)?
In version 3.0.2 the dependency biobb_common has been updated to 3.0.1 version.

### New features

* Update to biobb_common 3.0.1

## What's new in version [3.0.1](https://github.com/bioexcel/biobb_ml/releases/tag/v3.0.1)?
In version 3.0.1 TensorFlow, scikit-learn and pandas have been updated and fixed to the latest version of July 2020. Finally a new conda installation recipe has been introduced.

### New features

* Update to TensorFlow 2.2.0 (neural networks module)
* Update to scikit-learn 0.23.1 (classification, clustering, dimensionality_reduction and regression modules)
* Update to pandas 1.0.5 (general)
* New conda installer (installation)

## What's new in version [3.0.0](https://github.com/bioexcel/biobb_ml/releases/tag/v3.0.0)?
In version 3.0.0 Python has been updated to version 3.7. Big changes in the documentation style and content. Finally a new conda installation recipe has been introduced.

### New features

* Update to Python 3.7 (general)
* New conda installer (installation)
* Adding type hinting for easier usage (API)
* Deprecating os.path in favour of pathlib.path (modules)
* New command line documentation (documentation)

### Other changes

* New documentation styles (documentation)