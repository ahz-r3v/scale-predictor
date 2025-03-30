# test_grpc_client.py
import unittest
import grpc
import threading
import time

from src.scale_predictor import scale_predictor_pb2, scale_predictor_pb2_grpc, server
from src.scale_predictor import predictor

predictor.window_average = lambda index, window: sum(window) / len(window) if window else 0
predictor.trim_window = lambda index, window: window

class TestGRPCClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 50052
        cls.server_thread = threading.Thread(target=server.serve, args=("default", cls.port), daemon=True)
        cls.server_thread.start()
        time.sleep(1)
        cls.address = f'localhost:{cls.port}'

    def test_predict_with_invalid_function(self):
        channel = grpc.insecure_channel(self.address)
        stub = scale_predictor_pb2_grpc.ScalePredictorStub(channel)
        request = scale_predictor_pb2.PredictRequest(
            function_name="invalid_function",
            window=[1, 2, 3, 4, 5],
            index=0
        )
        response = stub.Predict(request)
        expected = sum([1, 2, 3, 4, 5]) / 5
        self.assertAlmostEqual(response.result, expected, places=6)

    def test_predict_with_valid_function(self):
        channel = grpc.insecure_channel(self.address)
        stub = scale_predictor_pb2_grpc.ScalePredictorStub(channel)

        request = scale_predictor_pb2.PredictRequest(
            function_name="trace-func-1",
            window=[2, 3, 4, 5, 6],
            index=0
        )
        response = stub.Predict(request)
        expected = sum([2, 3, 4, 5, 6]) / 5
        self.assertAlmostEqual(response.result, expected, places=6)

if __name__ == '__main__':
    unittest.main()
