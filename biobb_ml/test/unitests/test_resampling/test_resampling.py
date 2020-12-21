from biobb_common.tools import test_fixtures as fx
from biobb_ml.resampling.resampling import Resampling


class TestResampling():
    def setUp(self):
        fx.test_setup(self,'resampling')

    def tearDown(self):
        fx.test_teardown(self)
        pass

    def test_resampling(self):
        Resampling(properties=self.properties, **self.paths).launch()
        assert fx.not_empty(self.paths['output_dataset_path'])
        #assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])