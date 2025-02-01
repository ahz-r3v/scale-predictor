import unittest
import grpc
from concurrent import futures
import time
import socket
import math


# Import the generated classes
import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc

from src.scale_predictor.server import ScalePredictorService


def get_free_port():
    """
    Helper function to find a free port on the host.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class TestScalePredictorService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the gRPC server and client before any tests run.
        """
        # Get a free port
        cls.port = get_free_port()

        # Create a gRPC server
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # Instantiate the service
        cls.service = ScalePredictorService()

        # Add the service to the server
        scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
            cls.service,
            cls.server
        )

        # Add a port to the server
        cls.server.add_insecure_port(f'[::]:{cls.port}')

        # Start the server
        cls.server.start()

        # Allow some time for the server to start
        time.sleep(1)

        # Create a channel and stub
        cls.channel = grpc.insecure_channel(f'localhost:{cls.port}')
        cls.stub = scale_predictor_pb2_grpc.ScalePredictorStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        """
        Tear down the gRPC server and client after all tests run.
        """
        cls.server.stop(0)
        cls.channel.close()

    def test_predict_success(self):
        """
        Test that Predict returns the correct result for a trained function.
        """
        # Prepare and train the predictor
        dataset = {
            "funcA": [1, 2, 3, 4, 5],
        }
        window_size = 2
        self.service.predictor.train(dataset, window_size)

        # Define the window and index
        window = [4, 5]
        index = 1

        # Make the Predict request
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcA",
            window=window,
            index=index
        )
        response = self.stub.Predict(request)

        # Compute expected prediction
        expected_prediction = math.ceil(self.service.predictor.models["funcA"].predict([window])[0])
        expected_prediction = max(expected_prediction, 0)

        # Assert the response
        self.assertEqual(response.result, expected_prediction)

    def test_predict_not_trained(self):
        """
        Test that Predict raises an error when the predictor has not been trained.
        """
        # Attempt to predict without training
        window = [1, 2]
        index = 1
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcA",
            window=window,
            index=index
        )
        with self.assertRaises(grpc.RpcError) as context:
            self.stub.Predict(request)

        self.assertEqual(context.exception.code(), grpc.StatusCode.INTERNAL)
        self.assertIn("No trained model found for function 'funcA'", context.exception.details())

    def test_predict_unregistered_function(self):
        """
        Test that Predict raises an error when requesting a prediction for an unregistered function.
        """
        # Prepare and train the predictor with a different function
        dataset = {
            "funcA": [1, 2, 3, 4, 5],
        }
        window_size = 2
        self.service.predictor.train(dataset, window_size)

        # Attempt to predict for an unregistered function
        window = [1, 2]
        index = 1
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcB",
            window=window,
            index=index
        )
        with self.assertRaises(grpc.RpcError) as context:
            self.stub.Predict(request)

        # Check that the error code is INTERNAL
        self.assertEqual(context.exception.code(), grpc.StatusCode.INTERNAL)
        self.assertIn("No trained model found for function 'funcB'", context.exception.details())

    def test_predict_negative_result(self):
        """
        Test that Predict returns 0 when the prediction is negative.
        """
        # Prepare and train the predictor with data that may lead to negative predictions
        dataset = {
            "funcA": [10, 0, -10, -20, -30],
        }
        window_size = 3
        self.service.predictor.train(dataset, window_size)

        # Define the window and index
        window = [-20, -30, -40]
        index = 2

        # Make the Predict request
        request = scale_predictor_pb2.PredictRequest(
            function_name="funcA",
            window=window,
            index=index
        )
        response = self.stub.Predict(request)

        # Since prediction is negative, expect 0
        expected_prediction = 0
        self.assertEqual(response.result, expected_prediction)

if __name__ == '__main__':
    unittest.main()
