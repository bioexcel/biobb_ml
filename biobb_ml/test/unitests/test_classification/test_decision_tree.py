from biobb_common.tools import test_fixtures as fx
from biobb_ml.classification.decision_tree import DecisionTree


class TestDecisionTree():
    def setUp(self):
        fx.test_setup(self,'decision_tree')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_decision_tree(self):
        DecisionTree(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_model_path'])
        assert fx.equal(self.paths['output_model_path'], self.paths['ref_output_model_path'])
        assert fx.not_empty(self.paths['output_test_table_path'])
        assert fx.equal(self.paths['output_test_table_path'], self.paths['ref_output_test_table_path'])
        assert fx.not_empty(self.paths['output_plot_path'])
        assert fx.equal(self.paths['output_plot_path'], self.paths['ref_output_plot_path'])