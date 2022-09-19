from biobb_common.tools import test_fixtures as fx
from biobb_ml.classification.support_vector_machine import support_vector_machine


class TestSupportVectorMachine():
    def setup_class(self):
        fx.test_setup(self,'support_vector_machine')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_support_vector_machine(self):
        support_vector_machine(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_path'])
        #assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_table_path'])
        #assert fx.equal(self.paths['output_test_table_path'], self.paths['ref_output_test_table_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        #assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
