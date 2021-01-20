from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.map_variables import map_variables


class TestMapVariables():
    def setUp(self):
        fx.test_setup(self,'map_variables')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_map_variables(self):
        map_variables(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_dataset_path'])
        assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])