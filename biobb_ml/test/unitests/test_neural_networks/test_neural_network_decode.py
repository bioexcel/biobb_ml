from biobb_common.tools import test_fixtures as fx
from biobb_ml.neural_networks.neural_network_decode import neural_network_decode


class TestDecodingNeuralNetwork():
    def setup_class(self):
        fx.test_setup(self,'neural_network_decode')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_neural_network_decode(self):
        neural_network_decode(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_decode_path'])
        assert fx.equal(self.paths['output_decode_path'], self.paths['ref_output_decode_path'])
        assert fx.not_empty(self.paths['output_predict_path'])
        assert fx.equal(self.paths['output_predict_path'], self.paths['ref_output_predict_path'])