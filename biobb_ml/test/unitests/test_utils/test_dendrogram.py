from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.dendrogram import dendrogram
from biobb_ml.test.unitests.common import compare_images


class TestDendrogram():
    def setup_class(self):
        fx.test_setup(self, 'dendrogram')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_dendrogram(self):
        dendrogram(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_plot_path'])
        assert compare_images(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
