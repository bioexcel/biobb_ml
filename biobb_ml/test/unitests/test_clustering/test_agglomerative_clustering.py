from biobb_common.tools import test_fixtures as fx
from biobb_ml.clustering.agglomerative_clustering import agglomerative_clustering
from biobb_ml.test.unitests.common import compare_images


class TestAgglClustering():
    def setup_class(self):
        fx.test_setup(self, 'agglomerative_clustering')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_agglomerative_clustering(self):
        agglomerative_clustering(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert compare_images(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
