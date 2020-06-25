from biobb_common.tools import test_fixtures as fx
from biobb_ml.dimensionality_reduction.pls_components import PLSComponents


class TestPLSComponents():
    def setUp(self):
        fx.test_setup(self,'pls_components')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_pls_components(self):
        PLSComponents(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])