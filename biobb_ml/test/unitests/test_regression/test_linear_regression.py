from biobb_common.tools import test_fixtures as fx
from biobb_ml.regression.linear_regression import linear_regression


class TestLinearRegression():
    def setUp(self):
        fx.test_setup(self,'linear_regression')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_linear_regression(self):
        linear_regression(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_path'])
        assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_table_path'])
        assert fx.equal(self.paths['output_test_table_path'], self.paths['ref_output_test_table_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
