import unittest
import pandas as pd
import os

# 假设 generate_dataframe 函数在 data_processor.py 文件中
from src.scale_predictor.nhits import NHITSModel  

class TestGenerateDataFrame(unittest.TestCase):

    def setUp(self):

        self.test_file_path = "data/testout.csv"
        self.train_file = "data/train.csv"
        self.test_file = "data/test.csv"

        # 初始化 NHITSModel
        self.model = NHITSModel(60)

    def test_generate_dataframe(self):
        """ 测试 generate_dataframe 生成的数据是否符合预期 """

        # 运行数据处理函数
        train_df, test_df = self.model.generate_dataframe(self.test_file_path, self.train_file, self.test_file)

        # 预期列
        expected_columns = {'unique_id', 'timestamp', 'y', 'ds'}
        self.assertTrue(expected_columns.issubset(train_df.columns), "训练集列名错误")
        self.assertTrue(expected_columns.issubset(test_df.columns), "测试集列名错误")

        # 针对每个 unique_id，检查时间戳完整性
        for uid in train_df['unique_id'].unique():
            uid_train = train_df[train_df['unique_id'] == uid]
            uid_test = test_df[test_df['unique_id'] == uid]

            # 计算 unique_id 的完整时间范围
            min_time = uid_train['timestamp'].min()
            max_time = uid_test['timestamp'].max()
            expected_timestamps = list(range(min_time, max_time + 1))

            # 合并训练集和测试集，检查时间戳是否完整
            actual_timestamps = pd.concat([uid_train, uid_test])['timestamp'].tolist()
            self.assertEqual(expected_timestamps, actual_timestamps, f"时间戳未正确填充 for unique_id {uid}")

        # 检查 `cpu` 是否正确填充为 `y`
        missing_rows = pd.concat([train_df, test_df])[['unique_id', 'timestamp', 'y']].isna().sum().sum()
        self.assertEqual(missing_rows, 0, "y 值未正确填充")

        # 训练集和测试集划分是否正确（80%-20%）
        for uid in train_df['unique_id'].unique():
            uid_train = train_df[train_df['unique_id'] == uid]
            uid_test = test_df[test_df['unique_id'] == uid]

            total_time_steps = len(uid_train['timestamp'].unique()) + len(uid_test['timestamp'].unique())
            expected_train_size = int(0.8 * total_time_steps)

            self.assertEqual(len(uid_train['timestamp'].unique()), expected_train_size, f"训练集大小错误 for unique_id {uid}")

        print("✅ 测试通过：数据格式正确！")

    def test_train_and_predict(self):
        """ 测试 NHITS 训练和预测 """
        
        # 训练模型
        self.model.train_from_file(self.train_file)

        df = pd.read_csv(self.train_file)
        window = df[df['unique_id'] == 31]['y'].iloc[:60]
        print(window)

        # 执行预测
        predicted_value = self.model.predict('31', window)

        # 确保预测值存在且合理
        self.assertIsNotNone(predicted_value, "❌ 预测值为空")

        print(f"✅ NHITS 预测成功：{predicted_value}")
    


    def tearDown(self):
        """ 清理测试数据 """
        # if os.path.exists("train.csv"):
        #     os.remove("train.csv")
        # if os.path.exists("test.csv"):
        #     os.remove("test.csv")

if __name__ == "__main__":
    unittest.main()
