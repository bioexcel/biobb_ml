from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.dummy_variables import DummyVariables


class TestDummyVariables():
    def setUp(self):
        fx.test_setup(self,'dummy_variables')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_dummy_variables(self):
        DummyVariables(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_dataset_path'])
        assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])