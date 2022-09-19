from biobb_common.tools import test_fixtures as fx
from biobb_ml.resampling.oversampling import oversampling


class TestOversampling():
    def setup_class(self):
        fx.test_setup(self,'oversampling')

    def teardown_class(self):
        fx.test_teardown(self)
        pass

    def test_oversampling(self):
        oversampling(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_dataset_path'])
        assert fx.equal(self.paths['output_dataset_path'], self.paths['ref_output_dataset_path'])