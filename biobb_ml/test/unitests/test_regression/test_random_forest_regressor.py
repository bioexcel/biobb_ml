from biobb_common.tools import test_fixtures as fx
from biobb_ml.regression.random_forest_regressor import random_forest_regressor
import platform


class TestRandomForestRegressor():
    def setup_class(self):
        fx.test_setup(self, 'random_forest_regressor')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_random_forest_regressor(self):
        random_forest_regressor(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_path'])
        if platform.system() == 'Darwin':
            assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_table_path'])
        assert fx.equal(self.paths['output_test_table_path'], self.paths['ref_output_test_table_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        # assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
