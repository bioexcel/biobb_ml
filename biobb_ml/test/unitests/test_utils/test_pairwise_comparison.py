from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.pairwise_comparison import pairwise_comparison


class TestPairwiseComparison():
    def setup_class(self):
        fx.test_setup(self, 'pairwise_comparison')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_pairwise_comparison(self):
        pairwise_comparison(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_plot_path'])
        # assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'], percent_tolerance=10)
