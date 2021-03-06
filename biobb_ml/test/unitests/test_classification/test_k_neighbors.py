from biobb_common.tools import test_fixtures as fx
from biobb_ml.classification.k_neighbors import k_neighbors


class TestKNeighborsTrain():
    def setUp(self):
        fx.test_setup(self,'k_neighbors')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_k_neighbors(self):
        k_neighbors(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_path'])
        assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_table_path'])
        assert fx.equal(self.paths['output_test_table_path'], self.paths['ref_output_test_table_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
