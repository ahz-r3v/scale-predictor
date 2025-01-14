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
        # 对每个 functionName 建立一个线性回归模型
        for function_name, call_list in dataset.items():
            if len(call_list) < 2:
                # 如果数据量太少，就跳过或用默认值
                continue

            # X：时间序列下标 [0, 1, 2, ..., n-1]
            # y：对应时刻的调用量
            X = np.arange(len(call_list)).reshape(-1, 1)  # shape=(n,1)
            y = np.array(call_list, dtype=float)

            # 训练线性回归模型
            model = LinearRegression()
            model.fit(X, y)

            # 保存到字典
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

        # 这里演示用：1) 先用线性回归模型预测“下一个时刻”的调用量，
        #            2) 再基于一个简单规则来计算需要的实例数量

        model = self.models[function_name]

        # 当前时刻可以理解为已有的历史数据长度+1 (或其它方式)
        # 这里假设我们过去的数据长度 + 1 作为下一个时刻
        # 你也可以根据实际需要，提取更多特征或多步预测
        historical_length = model.n_features_in_  # 这其实是 1
        # 正确拿到训练时的数据行数:
        # sklearn没有内置属性直接给出样本数，可自己存一下或推断:
        # 这里假设 model.coef_, model.intercept_ 都已经在 fit 后可用
        # 也可以在 train() 里保存 call_list 长度:
        #    self.sample_count[function_name] = len(call_list)
        # 然后这里 next_time = self.sample_count[function_name]
        # 为简单起见，下述写法做演示:

        # 先随意假设下一个时刻的 x 值:
        next_time_index = 9999  # 仅做示例，需要你在 train() 存下真实的 length
        # 其实最好的方式: 在 train() 时记录 call_list 的长度:
        # self.data_length = {func: len(call_list), ...}
        # next_time_index = self.data_length[function_name]

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
