from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.drop_columns import drop_columns


class TestDropColumns():
    def setup_class(self):
        fx.test_setup(self,'drop_columns')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_drop_columns(self):
        drop_columns(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_dataset_path'])
        assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])