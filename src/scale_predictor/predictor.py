import numpy as np
from typing import Dict, List
from sklearn.linear_model import LinearRegression

class ScalePredictor:
    """
    A predictor that, given historical trace data (for multiple functions),
    can be trained using linear regression to predict how many instances
    should be scaled at a certain time.

    Methods:
        train(dataset, window_size):
            Train a predictive model using historical data.
        predict(window) -> int:
            Predict the number of instances needed based on current usage window.
        clean():
            Clean or reset the current predictive model.
    """
    WINDOW_SIZE = 60

    def __init__(self):
        self.models: Dict[str, LinearRegression] = {}
        self.trained: bool = False

    def train(self, dataset: Dict[str, List[int]], window_size: int):
        """
        Train a linear regression model for each function based on its historical data.
        
        Args:
            dataset: 
                A dict: {functionName: [invocation data per sec, ...]}
            window_size:
                The window size (in seconds) you'd like to focus on, 
                or could be used for feature extraction. 
        """
        for function_name, call_list in dataset.items():
            if len(call_list) < 2:
                # If not enough data, skip and use default value.
                continue

            # X：[0, 1, 2, ..., n-1]
            # y：invocations
            X = np.arange(len(call_list)).reshape(-1, 1)  # shape=(n,1)
            y = np.array(call_list, dtype=float)

            # Linear Regression.
            model = LinearRegression()
            model.fit(X, y)

            self.models[function_name] = model

        self.trained = len(self.models) > 0

    def predict(self, function_name: str, window: List[int]) -> int:
        """
        Predict how many instances are needed for a given function based on
        the most recent usage window (e.g., last 60 seconds).

        Args:
            function_name (str): The name of the function to predict for.
            window (List[int]): Invocation data in the last minute (or any timespan).
        
        Returns:
            int: The number of instances needed (a simple example logic).
        """
        if not self.trained or function_name not in self.models:
            raise RuntimeError(
                f"No trained model found for function '{function_name}'. "
                "Make sure to call train() first and pass correct function_name."
            )

        model = self.models[function_name]

        # Next time is always WINDOWSIZE (second).
        next_time_index = len(window)

        # 取窗口的平均值(或者其它统计信息)，可以作为某种“增量”来修正
        current_avg = np.mean(window) if len(window) > 0 else 0.0

        # 线性回归预测下一个时刻调用量
        predicted_next = model.predict([[next_time_index]])[0]

        # 简单合并：假设当前窗口平均值 + 线性回归预测值融合
        # 这里仅作演示，不是严格的算法
        final_estimated_calls = (predicted_next + current_avg) / 2.0

        # 假设每个实例可以承载 10 次调用（举例而已）
        capacity_per_instance = 1
        needed_instances = max(1, int(np.ceil(final_estimated_calls / capacity_per_instance)))
        return needed_instances

    def clear(self):
        """
        Clean or reset all trained models, so we can retrain from scratch.
        """
        self.models.clear()
        self.trained = False
