from biobb_common.tools import test_fixtures as fx
from biobb_ml.utils.scale_columns import ScaleColumns


class TestScaleColumns():
    def setUp(self):
        fx.test_setup(self,'scale_columns')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_scale_columns(self):
        ScaleColumns(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_dataset_path'])
        assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])