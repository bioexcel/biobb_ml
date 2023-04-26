from biobb_common.tools import test_fixtures as fx
from biobb_ml.neural_networks.recurrent_neural_network import recurrent_neural_network


class TestRecurrentNeuralNetwork():
    def setup_class(self):
        fx.test_setup(self, 'recurrent_neural_network')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_recurrent_neural_network(self):
        recurrent_neural_network(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_path'])
        # assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_table_path'])
        # assert fx.equal(self.paths['output_test_table_path'], self.paths['ref_output_test_table_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'], percent_tolerance=10)
