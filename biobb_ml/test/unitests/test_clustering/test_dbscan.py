from biobb_common.tools import test_fixtures as fx
from biobb_ml.clustering.dbscan import dbscan
import platform

class TestDBSCANClustering():
    def setup_class(self):
        fx.test_setup(self,'dbscan')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_dbscan(self):
        dbscan(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        if platform.system() == 'Darwin':
            assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])