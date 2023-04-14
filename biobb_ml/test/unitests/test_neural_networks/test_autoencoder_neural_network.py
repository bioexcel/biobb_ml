from biobb_common.tools import test_fixtures as fx
from biobb_ml.neural_networks.autoencoder_neural_network import autoencoder_neural_network


class TestAutoencoderNeuralNetwork():
    def setup_class(self):
        fx.test_setup(self, 'autoencoder_neural_network')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_autoencoder_neural_network(self):
        autoencoder_neural_network(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_path'])
        # assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_decode_path'])
        # assert fx.equal(self.paths['output_test_decode_path'], self.paths['ref_output_test_decode_path'])
        assert fx.not_empty(self.paths['output_test_predict_path'])
        # assert fx.equal(self.paths['output_test_predict_path'], self.paths['ref_output_test_predict_path'])
