import unittest
import os
import pandas as pd
import time
import os
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
import polars as pl

from eva_utils import NHITSModel, LinearModel

class TestTrainNHITSModel(unittest.TestCase):
    def setUp(self):
        self.nhits_model = NHITSModel(window_size=600)
        self.lr_model = LinearModel(window_size=600)
        self.test_dir = os.path.dirname(__file__)
        
        self.t50_test = os.path.join(self.test_dir, "data/50_test.csv")
        self.t50_train = os.path.join(self.test_dir, "data/50_train.csv")
        self.t50_val = os.path.join(self.test_dir, "data/50_val.csv")

        self.warm10_test = os.path.join(self.test_dir, "data/warm10_test.csv")
        self.top10_test = os.path.join(self.test_dir, "data/top10_test.csv")

        self.full_test = os.path.join(self.test_dir, "../data/sets/full_test.csv")

        self.t50_shift_test = os.path.join(self.test_dir, "data/50_test_shift.csv")

        self.t400_shift_test = os.path.join(self.test_dir, "data/400_test_shift.csv")


    def tearDown(self):
        pass


    def test_load_and_evaluate_nhits(self):
        self.nhits_model = NHITSModel(window_size=600)
        success, trained_funcs = self.nhits_model.load_model()
        self.assertTrue(success)
        # self.assertEqual(set(trained_funcs), {"global"})
        for func in trained_funcs:
            self.assertIn(func, self.nhits_model.models)
        # self.nhits_model.evaluate_model_batching("global_mix_loss_fn_lr=1e-3", self.full_test)
        self.nhits_model.evaluate_model_batching("global_mixed_loss", self.t50_shift_test)
        # self.nhits_model.evaluate_model_batching("global", self.top10_test)

    # def test_load_and_evaluate_linear(self):
    #     self.lr_model = LinearModel(window_size=600)
    #     success, trained_funcs = self.lr_model.load_model()
    #     self.assertTrue(success)
    #     # self.assertEqual(set(trained_funcs), {"global"})
    #     for func in trained_funcs:
    #         self.assertIn(func, self.lr_model.models)
    #     # self.nhits_model.evaluate_model_batching("global_mix_loss_fn_lr=1e-3", self.full_test)
    #     self.lr_model.evaluate_model_batching("global_400_b10_lr", self.warm10_test)
    #     # self.nhits_model.evaluate_model_batching("global", self.top10_test)


if __name__ == '__main__':
    unittest.main()