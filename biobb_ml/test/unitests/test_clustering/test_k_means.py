from biobb_common.tools import test_fixtures as fx
from biobb_ml.clustering.k_means import k_means


class TestKMeansClustering():
    def setUp(self):
        fx.test_setup(self,'k_means')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_k_means(self):
        k_means(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        #assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_model_path'])
        #assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        #assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])