from biobb_common.tools import test_fixtures as fx
from biobb_ml.dimensionality_reduction.pls_components import pls_components
import platform


class TestPLSComponents():
    def setup_class(self):
        fx.test_setup(self, 'pls_components')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_pls_components(self):
        pls_components(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        if platform.system() == 'Darwin':
            assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
