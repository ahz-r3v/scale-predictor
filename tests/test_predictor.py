# test_predict.py
import unittest
from src.scale_predictor.predictor import ScalePredictor

# 定义简单的 dummy 实现
def dummy_window_average(index, window):
    return sum(window) / len(window) if window else 0

def dummy_trim_window(index, window):
    return window

# Dummy 模型用于 default 分支，simulate 线性回归的 predict 方法
class DummyDefaultModel:
    def predict(self, X):
        # 返回第一个特征向量各项之和作为预测值
        return [sum(X[0])]

# Dummy NHITS 模型，返回固定预测结果
class DummyNHITSModel:
    def predict(self, func_name, window):
        return True, 999

class TestScalePredictor(unittest.TestCase):
    def setUp(self):
        # 对 predictor 模块中工具函数进行 monkey-patch
        from src.scale_predictor import predictor as predictor_mod
        predictor_mod.window_average = dummy_window_average
        predictor_mod.trim_window = dummy_trim_window

    def test_predict_default_fallback(self):
        # 当函数名不符合正则时，应使用 window_average 计算预测值
        predictor_instance = ScalePredictor(model_selector="default")
        # 未训练，模型中没有对应项
        result = predictor_instance.predict("invalid_func", [1, 2, 3, 4, 5], index=0)
        expected = dummy_window_average(0, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, expected, places=6)

    def test_predict_default_model(self):
        # 测试 default 模型时，提前设置 trained 状态，并手动添加 dummy 模型
        predictor_instance = ScalePredictor(model_selector="default")
        predictor_instance.trained = True
        # 注意：predict 中正则提取 "trace-func-1" 得到 key "1"
        predictor_instance.models["1"] = DummyDefaultModel()
        result = predictor_instance.predict("trace-func-1", [1, 2, 3, 4, 5], index=0)
        # DummyDefaultModel 返回的预测值为 1+2+3+4+5 = 15
        self.assertEqual(result, 15)

    def test_predict_nhits(self):
        # 测试 nhits 模型分支
        predictor_instance = ScalePredictor(model_selector="nhits")
        predictor_instance.trained = True
        # 模拟 nhits 模型的预测，key 同样为 "1"
        predictor_instance.models["1"] = DummyNHITSModel()
        # 构造一个长度等于 nhits 模型 window_size 的窗口
        window = list(range(1, predictor_instance.nhitsmdl.window_size + 1))
        result = predictor_instance.predict("trace-func-1", window, index=0)
        # DummyNHITSModel 固定返回 999
        self.assertEqual(result, 999)

    def test_predict_no_trained_model(self):
        # 当模型未训练时，无论函数名如何，都应回退使用 window_average
        predictor_instance = ScalePredictor(model_selector="default")
        predictor_instance.trained = False
        result = predictor_instance.predict("trace-func-1", [2, 4, 6, 8], index=0)
        expected = dummy_window_average(0, [2, 4, 6, 8])
        self.assertAlmostEqual(result, expected, places=6)

if __name__ == '__main__':
    unittest.main()
