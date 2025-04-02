import unittest
import os
import tempfile
import pandas as pd

from train_NHiTS import NHITSModel

class TestTrainNHITSModel(unittest.TestCase):
    def setUp(self):
        self.nhits_model = NHITSModel(window_size=600)
        self.test_dir = os.path.dirname(__file__)
        
        self.t50_test = os.path.join(self.test_dir, "data/50_test.csv")
        self.t50_train = os.path.join(self.test_dir, "data/50_train.csv")
        self.t50_val = os.path.join(self.test_dir, "data/50_val.csv")

        self.warm10_test = os.path.join(self.test_dir, "data/warm10_test.csv")
        self.top10_test = os.path.join(self.test_dir, "data/top10_test.csv")


    def tearDown(self):
        pass


    def test_load_and_evaluate_nhits(self):
        self.nhits_model = NHITSModel(window_size=600)
        success, trained_funcs = self.nhits_model.load_model()
        self.assertTrue(success)
        # self.assertEqual(set(trained_funcs), {"global"})
        for func in trained_funcs:
            self.assertIn(func, self.nhits_model.models)
        self.nhits_model.evaluate_model_batching("global", self.t50_test)
        self.nhits_model.evaluate_model_batching("global", self.warm10_test)
        self.nhits_model.evaluate_model_batching("global", self.top10_test)



if __name__ == '__main__':
    unittest.main()