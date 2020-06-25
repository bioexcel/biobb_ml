from biobb_common.tools import test_fixtures as fx
from biobb_ml.neural_networks.neural_network_predict import PredictNeuralNetwork


class TestPredictNeuralNetwork():
    def setUp(self):
        fx.test_setup(self,'neural_network_predict')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_neural_network_predict(self):
        PredictNeuralNetwork(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_results_path'])
        assert fx.equal(self.paths['output_results_path'], self.paths['ref_output_results_path'])