import unittest
from src.scale_predictor.predictor import ScalePredictor

# Dummy 线性模型，模拟 predict 方法
class DummyLinearModel:
    def predict(self, func_name, window):
        return True, sum(window) * 0.1  # 模拟线性回归输出

# 定义简单的 dummy 实现

def dummy_window_average(index, window):
    return sum(window) / len(window) if window else 0

def dummy_trim_window(index, window):
    return window

class TestScalePredictorLinear(unittest.TestCase):
    def setUp(self):
        # 对 predictor 模块中工具函数进行 monkey-patch
        from src.scale_predictor import predictor as predictor_mod
        predictor_mod.window_average = dummy_window_average
        predictor_mod.trim_window = dummy_trim_window
    
    def test_predict_default_fallback(self):
        predictor_instance = ScalePredictor(model_selector="linear")
        result = predictor_instance.predict("invalid_func", [1, 2, 3, 4, 5], index=0)
        expected = dummy_window_average(0, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_predict_linear_model(self):
        predictor_instance = ScalePredictor(model_selector="linear")
        predictor_instance.trained = True
        predictor_instance.models["1"] = DummyLinearModel()
        result = predictor_instance.predict("trace-func-1", [1, 2, 3, 4, 5], index=0)
        expected = sum([1, 2, 3, 4, 5]) * 0.1
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_predict_no_trained_model(self):
        predictor_instance = ScalePredictor(model_selector="linear")
        predictor_instance.trained = False
        result = predictor_instance.predict("trace-func-1", [2, 4, 6, 8], index=0)
        expected = dummy_window_average(0, [2, 4, 6, 8])
        self.assertAlmostEqual(result, expected, places=6)

if __name__ == '__main__':
    unittest.main()
