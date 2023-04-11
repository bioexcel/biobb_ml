import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biobb_ml",
    version="4.0.0",
    author="Biobb developers",
    author_email="genis.bayarri@irbbarcelona.org",
    description="Biobb_ml is the Biobb module collection to perform machine learning predictions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Bioinformatics Workflows BioExcel Compatibility",
    url="https://github.com/bioexcel/biobb_ml",
    project_urls={
        "Documentation": "http://biobb_ml.readthedocs.io/en/latest/",
        "Bioexcel": "https://bioexcel.eu/"
    },
    packages=setuptools.find_packages(exclude=['docs', 'test']),
    install_requires=['biobb_common==4.0.0', 'scikit-learn ==0.24.2', 'pandas ==1.3.0', 'seaborn ==0.10.1', 'tensorflow>=2.4.2', 'h5py ==2.10.0', 'imbalanced-learn ==0.7.0'],
    python_requires='>=3.7,<3.10',
    entry_points={
        "console_scripts": [
            "classification_predict = biobb_ml.classification.classification_predict:main",
            "decision_tree = biobb_ml.classification.decision_tree:main",
            "k_neighbors_coefficient = biobb_ml.classification.k_neighbors_coefficient:main",
            "k_neighbors = biobb_ml.classification.k_neighbors:main",
            "logistic_regression = biobb_ml.classification.logistic_regression:main",
            "random_forest_classifier = biobb_ml.classification.random_forest_classifier:main",
            "support_vector_machine = biobb_ml.classification.support_vector_machine:main",
            "agglomerative_clustering = biobb_ml.clustering.agglomerative_clustering:main",
            "agglomerative_coefficient = biobb_ml.clustering.agglomerative_coefficient:main",
            "clustering_predict = biobb_ml.clustering.clustering_predict:main",
            "dbscan = biobb_ml.clustering.dbscan:main",
            "k_means_coefficient = biobb_ml.clustering.k_means_coefficient:main",
            "k_means = biobb_ml.clustering.k_means:main",
            "spectral_clustering = biobb_ml.clustering.spectral_clustering:main",
            "spectral_coefficient = biobb_ml.clustering.spectral_coefficient:main",
            "pls_components = biobb_ml.dimensionality_reduction.pls_components:main",
            "pls_regression = biobb_ml.dimensionality_reduction.pls_regression:main",
            "principal_component = biobb_ml.dimensionality_reduction.principal_component:main",
            "autoencoder_neural_network = biobb_ml.neural_networks.autoencoder_neural_network:main",
            "classification_neural_network = biobb_ml.neural_networks.classification_neural_network:main",
            "neural_network_decode = biobb_ml.neural_networks.neural_network_decode:main",
            "neural_network_predict = biobb_ml.neural_networks.neural_network_predict:main",
            "recurrent_neural_network = biobb_ml.neural_networks.recurrent_neural_network:main",
            "regression_neural_network = biobb_ml.neural_networks.regression_neural_network:main",
            "linear_regression = biobb_ml.regression.linear_regression:main",
            "polynomial_regression = biobb_ml.regression.polynomial_regression:main",
            "random_forest_regressor = biobb_ml.regression.random_forest_regressor:main",
            "regression_predict = biobb_ml.regression.regression_predict:main",
            "oversampling = biobb_ml.resampling.oversampling:main",
            "resampling = biobb_ml.resampling.resampling:main",
            "undersampling = biobb_ml.resampling.undersampling:main",
            "correlation_matrix = biobb_ml.utils.correlation_matrix:main",
            "dendrogram = biobb_ml.utils.dendrogram:main",
            "drop_columns = biobb_ml.utils.drop_columns:main",
            "dummy_variables = biobb_ml.utils.dummy_variables:main",
            "map_variables = biobb_ml.utils.map_variables:main",
            "pairwise_comparison = biobb_ml.utils.pairwise_comparison:main",
            "scale_columns = biobb_ml.utils.scale_columns:main"
        ]
    },
    classifiers=(
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ),
)
