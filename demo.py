from src.scale_predictor.predictor import ScalePredictor

if __name__ == "__main__":
    # 模拟训练集: 两个函数的历史调用
    dataset = {
        "funcA": [10, 12, 15, 18, 20, 26, 30, 28, 23, 0, 23, 54, 34, 4, 435, 54, 0, 0, 0, 34, 4, 3, 2, 5, 0, 1],
        "funcB": [2, 3, 5, 7, 9, 9, 10, 5, 34, 23, 5, 1, 1, 0]
    }
    predictor = ScalePredictor()

    # 训练
    predictor.train(dataset, window_size=5)

    # predict: 假设 "funcA" 最近一分钟调用量情况(60秒聚合或每秒? 这里简化)
    current_window_a = [35, 32, 31, 40, 38, 36]  # 最近 6 秒调用量例子
    needed_a = predictor.predict("funcA", current_window_a)
    print(f"Predicted needed instances for funcA = {needed_a}")

    # predict: 对 "funcB" 做相同预测
    current_window_b = [9, 12, 14, 8, 11]
    needed_b = predictor.predict("funcB", current_window_b)
    print(f"Predicted needed instances for funcB = {needed_b}")

    # 清除模型
    predictor.clear()
