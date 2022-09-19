from biobb_common.tools import test_fixtures as fx
from biobb_ml.regression.regression_predict import regression_predict


class TestRegressionPredict():
    def setup_class(self):
        fx.test_setup(self,'regression_predict')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_regression_predict(self):
        regression_predict(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])