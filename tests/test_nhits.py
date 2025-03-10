# test_nhits.py
import unittest
import os
import tempfile
import pandas as pd

from src.scale_predictor.nhits import NHITSModel

# 定义一个 DummyModel，用于模拟 predict 返回固定结果
class DummyModel:
    def predict(self, input_df):
        # 返回一个包含 NHITS 列的 DataFrame，值固定为 100
        return pd.DataFrame({'NHITS': [100]})

# 定义 dummy 的 train_one_model 方法，避免调用 neuralforecast 实际训练逻辑
def dummy_train_one_model(self, func_name, train):
    self.models[func_name] = DummyModel()

class TestNHITSModel(unittest.TestCase):
    def setUp(self):
        # 创建 NHITSModel 实例，window_size 取 5
        self.nhits_model = NHITSModel(window_size=5)
        # 替换 train_one_model 为 dummy 实现
        self.nhits_model.train_one_model = dummy_train_one_model.__get__(self.nhits_model, NHITSModel)

        # 创建临时 CSV 文件作为测试数据
        self.test_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.test_dir.name, "sample.csv")
        with open(self.csv_path, "w") as f:
            # 第一行为表头（实际被 skiprows=1 忽略）
            f.write("timestamp,function,cpu\n")
            # 为两个函数生成 10 条数据
            for t in range(1, 11):
                f.write(f"{t},trace-func-1,{t * 0.1}\n")
            for t in range(1, 11):
                f.write(f"{t},trace-func-2,{t * 0.2}\n")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_generate_dataframe(self):
        train_csv = os.path.join(self.test_dir.name, "train.csv")
        test_csv = os.path.join(self.test_dir.name, "test.csv")
        train_df, test_df = self.nhits_model.generate_dataframe(self.csv_path, train_csv, test_csv)
        # 检查生成的数据集非空
        self.assertFalse(train_df.empty)
        self.assertFalse(test_df.empty)
        # 检查文件是否生成
        self.assertTrue(os.path.exists(train_csv))
        self.assertTrue(os.path.exists(test_csv))
        # 检查生成的 DataFrame 包含 unique_id 和 ds 列，且 ds 为日期类型
        self.assertIn("unique_id", train_df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(train_df['ds']))

    def test_train_from_file(self):
        # 测试从 CSV 文件训练模型
        success, trained_funcs = self.nhits_model.train_from_file(self.csv_path, window_size=5)
        self.assertTrue(success)
        # CSV 中包含两个函数：trace-func-1 和 trace-func-2
        self.assertEqual(set(trained_funcs), {"trace-func-1", "trace-func-2"})
        # 检查模型字典中是否添加了相应的键
        for func in trained_funcs:
            self.assertIn(func, self.nhits_model.models)

    def test_predict(self):
        # 模拟训练完成，为 "trace-func-1" 赋予一个 DummyModel
        self.nhits_model.models["trace-func-1"] = DummyModel()
        window_valid = [1, 2, 3, 4, 5]
        # 对已存在的函数调用 predict
        success, prediction = self.nhits_model.predict("trace-func-1", window_valid)
        self.assertTrue(success)
        self.assertEqual(prediction, 100)

        # 对不存在的函数调用 predict，应返回 (False, 0)
        success, prediction = self.nhits_model.predict("trace-func-unknown", window_valid)
        self.assertFalse(success)
        self.assertEqual(prediction, 0)

        # 输入窗口长度不符时，预测应返回失败
        window_invalid = [1, 2, 3]
        success, prediction = self.nhits_model.predict("trace-func-1", window_invalid)
        self.assertFalse(success)
        self.assertEqual(prediction, 0)

if __name__ == '__main__':
    unittest.main()
