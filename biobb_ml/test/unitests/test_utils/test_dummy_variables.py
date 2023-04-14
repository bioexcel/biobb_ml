from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.dummy_variables import dummy_variables


class TestDummyVariables():
    def setup_class(self):
        fx.test_setup(self, 'dummy_variables')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_dummy_variables(self):
        dummy_variables(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_dataset_path'])
        assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])
