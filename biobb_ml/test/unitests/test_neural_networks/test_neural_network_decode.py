from biobb_common.tools import test_fixtures as fx
from biobb_ml.neural_networks.neural_network_decode import DecodingNeuralNetwork


class TestDecodingNeuralNetwork():
    def setUp(self):
        fx.test_setup(self,'neural_network_decode')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_neural_network_decode(self):
        DecodingNeuralNetwork(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_decode_path'])
        assert fx.equal(self.paths['output_decode_path'], self.paths['ref_output_decode_path'])
        assert fx.not_empty(self.paths['output_predict_path'])
        assert fx.equal(self.paths['output_predict_path'], self.paths['ref_output_predict_path'])