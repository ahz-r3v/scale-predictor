import unittest
import os
import tempfile
import pandas as pd

from src.scale_predictor.nhits import NHITSModel

class TestTrainNHITSModel(unittest.TestCase):
    def setUp(self):
        self.nhits_model = NHITSModel(window_size=600)
        self.test_dir = os.path.dirname(__file__)

        self.t50_test = os.path.join(self.test_dir, "data/50_test.csv")
        self.t50_train = os.path.join(self.test_dir, "data/50_train.csv")
        self.t50_val = os.path.join(self.test_dir, "data/50_val.csv")
        self.t400_val = os.path.join(self.test_dir, "data/400_val.csv")

        self.t50_train_d = os.path.join(self.test_dir, "data/50_train_d.csv")
        self.t400_train_d = os.path.join(self.test_dir, "data/400_train_d.csv")
        self.t400_val_d = os.path.join(self.test_dir, "data/400_val_d.csv")
        self.t400_test_d = os.path.join(self.test_dir, "data/400_test_d.csv")

        self.t400_top10_train_d = os.path.join(self.test_dir, "data/400+top10_train_d.csv")
        self.t400_top5_train_d = os.path.join(self.test_dir, "data/400+top5_train_d.csv")
        self.t400_top10_test_d = os.path.join(self.test_dir, "data/400+top10_test_d.csv")
        self.t400_top10_test_d1 = os.path.join(self.test_dir, "data/400+top10_test_d_1.csv")

        self.t400_shift_train = os.path.join(self.test_dir, "data/400_train_shift.csv")
        self.t400_shift_val = os.path.join(self.test_dir, "data/400_val_shift.csv")
        
    def tearDown(self):
        pass

    def test_train_from_file_nhits(self):
        success, trained_funcs = self.nhits_model.train_from_file(self.t400_shift_train, self.t400_shift_val, window_size=600)
        self.assertTrue(success)
        self.assertEqual(set(trained_funcs), {"global"})
        for func in trained_funcs:
            self.assertIn(func, self.nhits_model.models)



if __name__ == '__main__':
    unittest.main()