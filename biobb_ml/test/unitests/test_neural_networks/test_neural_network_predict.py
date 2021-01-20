from biobb_common.tools import test_fixtures as fx
from biobb_ml.neural_networks.neural_network_predict import neural_network_predict


class TestPredictNeuralNetwork():
    def setUp(self):
        fx.test_setup(self,'neural_network_predict')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_neural_network_predict(self):
        neural_network_predict(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])