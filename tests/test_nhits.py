# test_nhits.py
import unittest
import os
import tempfile
import pandas as pd

from src.scale_predictor.nhits import NHITSModel

class DummyModel:
    def predict(self, input_df):
        return pd.DataFrame({'NHITS': [100]})

def dummy_train_one_model(self, func_name, train):
    self.models[func_name] = DummyModel()

class TestNHITSModel(unittest.TestCase):
    def setUp(self):
        self.nhits_model = NHITSModel(window_size=600)
        # self.nhits_model.train_one_model = dummy_train_one_model.__get__(self.nhits_model, NHITSModel)
        self.test_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.test_dir.name, "sample.csv")
        with open(self.csv_path, "w") as f:
            f.write("timestamp,function,cpu\n")
            for t in range(1, 11):
                f.write(f"{t},trace-func-1,{t * 0.1}\n")
            for t in range(1, 11):
                f.write(f"{t},trace-func-2,{t * 0.2}\n")

    def tearDown(self):
        self.test_dir.cleanup()

    # def test_generate_dataframe(self):
    #     train_csv = os.path.join(self.test_dir.name, "train.csv")
    #     test_csv = os.path.join(self.test_dir.name, "test.csv")
    #     train_df, test_df = self.nhits_model.generate_dataframe(self.csv_path, train_csv, test_csv)
    #     self.assertFalse(train_df.empty)
    #     self.assertFalse(test_df.empty)
    #     self.assertTrue(os.path.exists(train_csv))
    #     self.assertTrue(os.path.exists(test_csv))
    #     self.assertIn("unique_id", train_df.columns)
    #     self.assertTrue(pd.api.types.is_datetime64_any_dtype(train_df['ds']))

    # def test_train_from_file(self):
    #     success, trained_funcs = self.nhits_model.train_from_file(self.csv_path, window_size=5)
    #     self.assertTrue(success)
    #     self.assertEqual(set(trained_funcs), {"trace-func-1", "trace-func-2"})
    #     for func in trained_funcs:
    #         self.assertIn(func, self.nhits_model.models)

    def test_predict(self):
        self.nhits_model.load_model()
        window_valid = list(range(1, 601))
        success, prediction = self.nhits_model.predict("trace-func-1", window_valid)
        self.assertTrue(success)
        # self.assertEqual(prediction, float)

        success, prediction = self.nhits_model.predict("trace-func-unknown", window_valid)
        self.assertTrue(success)
        # self.assertEqual(prediction, float)

        # window_invalid = [1, 2, 3]
        # success, prediction = self.nhits_model.predict("trace-func-1", window_invalid)
        # self.assertFalse(success)
        # self.assertEqual(prediction, 0)

if __name__ == '__main__':
    unittest.main()
