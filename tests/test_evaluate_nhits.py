import unittest
import os
import tempfile
import pandas as pd

from src.scale_predictor.nhits import NHITSModel

class TestTrainNHITSModel(unittest.TestCase):
    def setUp(self):
        self.nhits_model = NHITSModel(window_size=600)
        self.test_dir = os.path.dirname(__file__)
        self.t400_train_csv = os.path.join(self.test_dir, "400_train.csv")
        # self.full_trace_path = os.path.join(self.test_dir, "testdata/full.out")
        self.full_test_csv = os.path.join(self.test_dir, "full_test.csv")
        self.t400_test_csv = os.path.join(self.test_dir, "400_test.csv")
        self.top10_train_csv = os.path.join(self.test_dir, "top10_train_trimed.csv")
        self.top10_test_csv = os.path.join(self.test_dir, "top10_test_trimed.csv")
        self.top10_sec_train_csv = os.path.join(self.test_dir, "top10_functions_train.csv")
        self.top10_sec_test_csv = os.path.join(self.test_dir, "top10_functions_test.csv")
        self.warm10_test_trimed_csv = os.path.join(self.test_dir, "warm10_test_trimed.csv")
        self.warm10_train_trimed_csv =os.path.join(self.test_dir, "warm10_train_trimed.csv")

        self.t50_test = os.path.join(self.test_dir, "data/50_test.csv")
        self.t50_train = os.path.join(self.test_dir, "data/50_train.csv")
        self.t50_val = os.path.join(self.test_dir, "data/50_val.csv")

        self.top10_test = os.path.join(self.test_dir, "data/top10_test.csv")
        self.warm10_test = os.path.join(self.test_dir, "data/warm10_test.csv")

        self.t50_test_shift = os.path.join(self.test_dir, "data/50_test_shift.csv")
        

    def tearDown(self):
        pass


    def test_load_and_evaluate_linear(self):
        self.nhits_model = NHITSModel(window_size=600)
        success, trained_funcs = self.nhits_model.load_model()
        self.assertTrue(success)
        # self.assertEqual(set(trained_funcs), {"global"})
        for func in trained_funcs:
            self.assertIn(func, self.nhits_model.models)
        self.nhits_model.evaluate_model_batching("global_mix_loss_fn_edo", self.warm10_test)



if __name__ == '__main__':
    unittest.main()