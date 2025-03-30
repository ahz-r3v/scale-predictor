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
        
        self.top10_train_csv = os.path.join(self.test_dir, "top10_train.csv")
        self.top10_test_csv = os.path.join(self.test_dir, "top10_test.csv")
        self.top10_sec_train_csv = os.path.join(self.test_dir, "top10_functions_train_trimed.csv")
        self.top10_sec_test_csv = os.path.join(self.test_dir, "top10_functions_test_trimed.csv")
        
        self.t400_train_trimed_csv = os.path.join(self.test_dir, "data/400_train_trimed.csv")
        self.t400_test_trimed_csv = os.path.join(self.test_dir, "data/400_test_trimed.csv")
        self.warm10_test_trimed_csv = os.path.join(self.test_dir, "data/warm10_test_trimed.csv")
        self.warm10_train_trimed_csv = os.path.join(self.test_dir, "data/warm10_train_trimed.csv")

        self.t50_test = os.path.join(self.test_dir, "data/50_test.csv")
        self.t50_train = os.path.join(self.test_dir, "data/50_train.csv")
        self.t50_val = os.path.join(self.test_dir, "data/50_val.csv")
        self.t400_val = os.path.join(self.test_dir, "data/400_val.csv")

        self.t50_train_d = os.path.join(self.test_dir, "data/50_train_d.csv")
        self.t400_train_d = os.path.join(self.test_dir, "data/400_train_d.csv")
        self.t400_val_d = os.path.join(self.test_dir, "data/400_val_d.csv")
        self.t400_test_d = os.path.join(self.test_dir, "data/400_test_d.csv")
        
    def tearDown(self):
        pass

    def test_train_from_file_nhits(self):
        success, trained_funcs = self.nhits_model.train_from_file(self.t400_train_d, self.t400_test_d, window_size=600)
        self.assertTrue(success)
        self.assertEqual(set(trained_funcs), {"global"})
        for func in trained_funcs:
            self.assertIn(func, self.nhits_model.models)



if __name__ == '__main__':
    unittest.main()