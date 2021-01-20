from biobb_common.tools import test_fixtures as fx
from biobb_ml.dimensionality_reduction.pls_regression import pls_regression


class TestPLS_Regression():
    def setUp(self):
        fx.test_setup(self,'pls_regression')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_pls_regression(self):
        pls_regression(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])