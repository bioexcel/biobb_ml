from biobb_common.tools import test_fixtures as fx
from biobb_ml.clustering.spectral_coefficient import spectral_coefficient


class TestSpectralCoefficient():
    def setup_class(self):
        fx.test_setup(self, 'spectral_coefficient')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_spectral_coefficient(self):
        spectral_coefficient(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
