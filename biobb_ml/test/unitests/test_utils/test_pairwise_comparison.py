from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.pairwise_comparison import pairwise_comparison
from biobb_ml.test.unitests.common import compare_images


class TestPairwiseComparison():
    def setup_class(self):
        fx.test_setup(self, 'pairwise_comparison')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_pairwise_comparison(self):
        pairwise_comparison(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_plot_path'])
        assert compare_images(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
