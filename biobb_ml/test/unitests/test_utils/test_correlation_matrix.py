from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.correlation_matrix import correlation_matrix


class TestCorrelationMatrix():
    def setup_class(self):
        fx.test_setup(self,'correlation_matrix')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_correlation_matrix(self):
        correlation_matrix(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])