from biobb_common.tools import test_fixtures as fx
from biobb_ml.resampling.resampling import resampling


class TestResampling():
    def setup_class(self):
        fx.test_setup(self, 'resampling')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_resampling(self):
        resampling(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_dataset_path'])
        # assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])
