import unittest
import grpc
from concurrent import futures
import time
import socket
import math
import os
from src.scale_predictor.utils import window_average 

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc
from src.scale_predictor.server import ScalePredictorService

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

class TestScalePredictorService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        debug = os.getenv("PD_DEBUG", default='0')
        cls.port = get_free_port()
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        cls.service = ScalePredictorService(debug)
        scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
            cls.service,
            cls.server
        )
        cls.server.add_insecure_port(f'[::]:{cls.port}')
        cls.server.start()
        time.sleep(1)
        cls.channel = grpc.insecure_channel(f'localhost:{cls.port}')
        cls.stub = scale_predictor_pb2_grpc.ScalePredictorStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(0)
        cls.channel.close()

    def test_predict_success(self):
        dataset = {
            "funcA": [1, 2, 3, 4, 5],
        }
        window_size = 2
        self.service.predictor.train(dataset, window_size)
        window = [4, 5]
        index = 1
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcA",
            window=window,
            index=index
        )
        response = self.stub.Predict(request)
        expected_prediction = math.ceil(self.service.predictor.models["funcA"].predict([window])[0])
        expected_prediction = max(expected_prediction, 0)
        self.assertEqual(response.result, int(expected_prediction))

    def test_predict_not_trained(self):
        self.service.predictor.clear()
        window = [1, 2, 3]
        index = 1
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcA",
            window=window,
            index=index
        )
        response = self.stub.Predict(request)
        expected_prediction = window_average(index, window)
        self.assertEqual(response.result, int(expected_prediction))

    def test_predict_unregistered_function(self):
        dataset = {
            "funcA": [1, 2, 3, 4, 5],
        }
        window_size = 2
        self.service.predictor.train(dataset, window_size)
        window = [1, 2]
        index = 1
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcB",
            window=window,
            index=index
        )
        response = self.stub.Predict(request)
        expected_prediction = window_average(index, window)
        self.assertEqual(response.result, math.ceil(expected_prediction))

    def test_predict_negative_result(self):
        dataset = {
            "funcA": [10, 0, -10, -20, -30],
        }
        window_size = 3
        self.service.predictor.train(dataset, window_size)
        window = [-20, -30, -40]
        index = 2
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcA",
            window=window,
            index=index
        )
        response = self.stub.Predict(request)
        expected_prediction = 0
        self.assertEqual(response.result, int(expected_prediction))

if __name__ == '__main__':
    unittest.main()
