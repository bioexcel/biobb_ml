from biobb_common.tools import test_fixtures as fx
from biobb_ml.classification.classification_predict import classification_predict


class TestClassificationPredict():
    def setUp(self):
        fx.test_setup(self,'classification_predict')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_classification_predict(self):
        classification_predict(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])
