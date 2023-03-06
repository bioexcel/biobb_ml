from biobb_common.tools import test_fixtures as fx
from biobb_ml.classification.decision_tree import decision_tree
import platform

class TestDecisionTree():
    def setup_class(self):
        fx.test_setup(self,'decision_tree')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_decision_tree(self):
        decision_tree(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_path'])
        if platform.system() == 'Darwin': 
            assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_table_path'])
        assert fx.equal(self.paths['output_test_table_path'], self.paths['ref_output_test_table_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        if platform.system() == 'Darwin':
            assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])
