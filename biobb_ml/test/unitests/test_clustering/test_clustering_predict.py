from biobb_common.tools import test_fixtures as fx
from biobb_ml.clustering.clustering_predict import ClusteringPredict


class TestClusteringPredict():
    def setUp(self):
        fx.test_setup(self,'clustering_predict')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_clustering_predict(self):
        ClusteringPredict(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_results_path'])
        #assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])