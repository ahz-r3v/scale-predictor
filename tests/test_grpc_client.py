import unittest
import grpc
from concurrent import futures
import time
import socket
import math
import os
from src.scale_predictor.utils import window_average 
import logging

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
        model_selector = 'default'
        cls.port = get_free_port()
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        cls.service = ScalePredictorService(model_selector)
        scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
            cls.service,
            cls.server
        )
        cls.server.add_insecure_port(f'[::]:{cls.port}')
        cls.server.start()
        time.sleep(1)
        cls.channel = grpc.insecure_channel(f'localhost:{cls.port}')
        cls.stub = scale_predictor_pb2_grpc.ScalePredictorStub(cls.channel)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("predictor.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

    def setUpClassNhits(cls):
        model_selector = 'nhits'
        cls.port = get_free_port()
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        cls.service = ScalePredictorService(model_selector)
        scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
            cls.service,
            cls.server
        )
        cls.server.add_insecure_port(f'[::]:{cls.port}')
        cls.server.start()
        time.sleep(1)
        cls.channel = grpc.insecure_channel(f'localhost:{cls.port}')
        cls.stub = scale_predictor_pb2_grpc.ScalePredictorStub(cls.channel)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("predictor.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

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
        expected_prediction = self.service.predictor.models["funcA"].predict([window])[0]
        expected_prediction = max(expected_prediction, 0)
        self.assertEqual(response.result, expected_prediction)

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
        self.assertEqual(response.result, expected_prediction)

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

    def test_train_by_file(self):
        """
        Test TrainByFile gRPC method.
        """
        self.tearDownClass()
        self.setUpClassNhits()
        test_file_path = "data/testout.csv"
        window_size = 60

        def file_generator():
            with open(test_file_path, "rb") as f:
                while chunk := f.read(1024):
                    yield scale_predictor_pb2.FileChunk(
                        filename=test_file_path,
                        window_size=window_size,
                        data=chunk
                    )

        response = self.stub.TrainByFile(file_generator())

        self.assertTrue(response.success)
        self.assertIn("trained successfully", response.message)

        self.assertTrue(os.path.exists("./data/received.csv"))

        """
        Test gRPC Predict method with function_name="31"
        """
        window = [
                7.067220742927475, 7.357818507531192, 8.457729668753927, 9.03996365392004, 
                8.67572528623191, 7.272060673576107, 6.1308326460214175, 7.176282114926153, 
                6.439984927552587, 6.384000000000015, 5.464285403693566, 5.0, 
                5.322791718078861, 4.095157655053754, 4.387842344946193, 4.0, 
                4.0, 4.309172313154022, 3.9748336201064376, 4.096548978072178, 
                4.0, 4.226000000000113, 3.9465614795517463, 2.37434596074354, 
                2.102175530838167, 2.3900000000001, 2.371999999999844, 2.3990000000001146, 
                2.104938299533842, 2.3370617004661653, 1.3651814924689916, 1.5295190790475317, 
                1.381885239052508, 1.469000000000051, 1.0354061562593415, 1.504593843740622, 
                1.0, 1.0646506742382371, 1.150349325761681, 2.217643835512945, 
                2.0, 2.0, 2.0, 2.1269999999999527, 1.608970490547108, 1.378510430405413, 
                1.0, 1.4193800165182893, 2.0, 2.0, 2.3910000000000764, 2.0, 
                2.29099999999994, 2.0, 1.1453561644868842, 1.0, 1.0, 1.0, 
                1.35565989228121, 1.4194589725293554
            ]
        window_size = 60
        index = 0
        request = scale_predictor_pb2.PredictRequest(
            function_name="31",
            window=window,
            index=index
        )
        response = self.stub.Predict(request)
        print(response.result)



if __name__ == '__main__':
    unittest.main()
