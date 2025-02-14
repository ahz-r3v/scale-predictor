import unittest
import math
import os
import tempfile
from src.scale_predictor.predictor import ScalePredictor
from src.scale_predictor.utils import window_average  # 默认窗口平均算法

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = ScalePredictor('0')

    def test_train_no_data(self):
        """
        当输入数据为空时，trained 应该为 False。
        """
        dataset = {}
        window_size = 60
        self.predictor.train(dataset, window_size)

        self.assertFalse(self.predictor.trained, "`trained` should be False")
        self.assertEqual(len(self.predictor.models), 0, "empty dataset should not be used to train any model")

    def test_train_insufficient_data(self):
        """
        当某个函数的数据量少于2时，不生成模型。
        """
        dataset = {
            "funcA": [10],
            "funcB": [1, 2, 3, 4, 5]
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained)
        self.assertIn("funcB", self.predictor.models)
        self.assertNotIn("funcA", self.predictor.models)

    def test_train_and_predict_single_function(self):
        """
        对单个函数进行训练和预测，检查模型的可用性及预测输出。
        """
        dataset = {
            "funcA": [i for i in range(10)]  # 数据：0,1,2,...,9
        }
        window_size = 3
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained, "`trained` should be True")
        self.assertIn("funcA", self.predictor.models, "no model for funcA")

        window = [7, 8, 9] 
        predicted = self.predictor.predict("funcA", window, index=2)
        self.assertGreaterEqual(predicted, 0, f"Predicted value should be >= 0, but got {predicted}")

    def test_train_and_predict_multiple_functions(self):
        """
        多函数训练和预测。
        """
        dataset = {
            "funcA": [1, 2, 3, 4, 5, 6],
            "funcB": [10, 9, 8, 7, 6, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained, "`trained` should be True")
        self.assertIn("funcA", self.predictor.models)
        self.assertIn("funcB", self.predictor.models)

        # 对 funcA 预测
        windowA = [4, 5] 
        predA = self.predictor.predict("funcA", windowA, index=1)
        self.assertGreaterEqual(predA, 0)

        # 对 funcB 预测
        windowB = [6, 5] 
        predB = self.predictor.predict("funcB", windowB, index=1)
        self.assertGreaterEqual(predB, 0)

    def test_predict_without_training(self):
        """
        未训练情况下的预测不再抛异常，而是返回默认的 window_average 计算结果。
        """
        window = [1, 2, 3]
        index = 1
        expected = window_average(index, window)
        predicted = self.predictor.predict("notExistFunc", window, index)
        self.assertEqual(predicted, expected,
                         f"Expected default window_average value {expected} for untrained predictor, got {predicted}")

    def test_predict_unregistered_function(self):
        """
        未注册函数的预测返回默认值。
        """
        dataset = {
            "funcA": [1, 2, 3, 4],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)
        window = [1, 2]
        index = 1
        expected = math.ceil(window_average(index, window))
        predicted = self.predictor.predict("notExistFunc", window, index)
        self.assertEqual(predicted, expected,
                         f"Expected default window_average value {expected} for unregistered function, got {predicted}")

    def test_predict_negative_result(self):
        """
        当预测结果为负时，应返回 0。
        """
        dataset = {
            "funcA": [10, 0, -10, -20, -30, -40]
        }
        window_size = 3
        self.predictor.train(dataset, window_size)

        window = [-20, -30, -40]
        predicted = self.predictor.predict("funcA", window, 2)
        
        self.assertEqual(predicted, 0, f"Expected 0 for negative prediction, but got {predicted}")

    def test_clear(self):
        dataset = {
            "funcA": [1, 2, 3, 4, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        self.assertTrue(self.predictor.trained)
        self.assertIn("funcA", self.predictor.models)

        self.predictor.clear()

        self.assertFalse(self.predictor.trained)
        self.assertEqual(len(self.predictor.models), 0)
        self.assertEqual(self.predictor.window_size, 0)

    def test_export(self):
        """
        测试 export 方法是否正确保存状态到文件。
        """
        dataset = {
            "funcA": [1, 2, 3, 4, 5, 6],
            "funcB": [10, 9, 8, 7, 6, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        with tempfile.TemporaryDirectory() as tmpdirname:
            export_path = os.path.join(tmpdirname, 'scale_predictor.joblib')
            self.predictor.export(export_path)

            self.assertTrue(os.path.exists(export_path), "Export file was not created.")
            self.assertGreater(os.path.getsize(export_path), 0, "Export file is empty.")

    def test_load(self):
        """
        测试 load 方法是否正确恢复状态。
        """
        dataset = {
            "funcA": [1, 2, 3, 4, 5, 6],
            "funcB": [10, 9, 8, 7, 6, 5],
        }
        window_size = 2
        self.predictor.train(dataset, window_size)

        with tempfile.TemporaryDirectory() as tmpdirname:
            export_path = os.path.join(tmpdirname, 'scale_predictor.joblib')
            self.predictor.export(export_path)

            new_predictor = ScalePredictor('0')
            new_predictor.load(export_path)

            self.assertTrue(new_predictor.trained, "Loaded predictor should be trained.")
            self.assertEqual(new_predictor.window_size, self.predictor.window_size, "Window size mismatch after loading.")
            self.assertEqual(set(new_predictor.models.keys()), set(self.predictor.models.keys()),
                             "Function names mismatch after loading.")

            # 检查预测结果是否一致
            windowA = [4, 5] 
            predA_original = self.predictor.predict("funcA", windowA, index=1)
            predA_loaded = new_predictor.predict("funcA", windowA, index=1)
            self.assertEqual(predA_loaded, predA_original, "Predictions mismatch for funcA after loading.")

            windowB = [6, 5] 
            predB_original = self.predictor.predict("funcB", windowB, index=1)
            predB_loaded = new_predictor.predict("funcB", windowB, index=1)
            self.assertEqual(predB_loaded, predB_original, "Predictions mismatch for funcB after loading.")

    def test_export_and_load(self):
        """
        组合测试：导出、清除、加载后验证状态与预测结果。
        """
        dataset = {
            "funcA": [2, 4, 6, 8, 10, 12],
            "funcB": [12, 10, 8, 6, 4, 2],
        }
        window_size = 3
        self.predictor.train(dataset, window_size)

        with tempfile.TemporaryDirectory() as tmpdirname:
            export_path = os.path.join(tmpdirname, 'scale_predictor_combined.joblib')
            self.predictor.export(export_path)

            self.predictor.clear()
            self.assertFalse(self.predictor.trained)
            self.assertEqual(len(self.predictor.models), 0)
            self.assertEqual(self.predictor.window_size, 0)

            self.predictor.load(export_path)

            self.assertTrue(self.predictor.trained, "Predictor should be trained after loading.")
            self.assertEqual(self.predictor.window_size, window_size, "Window size mismatch after loading.")
            self.assertIn("funcA", self.predictor.models)
            self.assertIn("funcB", self.predictor.models)

            windowA = [6, 8, 10]
            predA = self.predictor.predict("funcA", windowA, index=2)
            expected_predA = math.ceil(self.predictor.models["funcA"].predict([windowA])[0])
            expected_predA = max(expected_predA, 0)
            self.assertEqual(predA, expected_predA, "Prediction for funcA mismatch after loading.")

            windowB = [8, 6, 4]
            predB = self.predictor.predict("funcB", windowB, index=2)
            expected_predB = math.ceil(self.predictor.models["funcB"].predict([windowB])[0])
            expected_predB = max(expected_predB, 0)
            self.assertEqual(predB, expected_predB, "Prediction for funcB mismatch after loading.")


if __name__ == '__main__':
    unittest.main()
