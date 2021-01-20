from biobb_common.tools import test_fixtures as fx
from biobb_ml.classification.k_neighbors_coefficient import k_neighbors_coefficient


class TestKNeighborsCoefficient():
    def setUp(self):
        fx.test_setup(self,'k_neighbors_coefficient')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_k_neighbors_coefficient(self):
        k_neighbors_coefficient(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
