import unittest
import grpc
import time
import multiprocessing
from concurrent import futures

import src.scale_predictor.scale_predictor_pb2 as pb2
import src.scale_predictor.scale_predictor_pb2_grpc as pb2_grpc

from main import run_grpc_server, HOST, PORT, NUM_PROCESSES 

class TestMultiProcessGRPCServer(unittest.TestCase):
    """Unit test for multi-process gRPC server with SO_REUSEPORT."""

    @classmethod
    def setUpClass(cls):
        """Start multiple gRPC server processes before tests."""
        cls.model_selector = "default"

        cls.processes = []
        for _ in range(NUM_PROCESSES):
            p = multiprocessing.Process(target=run_grpc_server, args=(cls.model_selector,))
            p.start()
            cls.processes.append(p)

        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        """Terminate gRPC server processes after tests."""
        for p in cls.processes:
            p.terminate()
            p.join()

    def test_server_health(self):
        """Test if gRPC server is responding."""
        with grpc.insecure_channel(f"{HOST}:{PORT}") as channel:
            stub = pb2_grpc.ScalePredictorStub(channel)
            request = pb2.PredictRequest(function_name="test_func", window=[0.1, 0.2, 0.3], index=1)
            response = stub.Predict(request)
            self.assertGreaterEqual(response.result, 0)

    def test_multiple_requests(self):
        """Test multiple gRPC requests in parallel."""
        num_requests = 20

        def grpc_request():
            with grpc.insecure_channel(f"{HOST}:{PORT}") as channel:
                stub = pb2_grpc.ScalePredictorStub(channel)
                request = pb2.PredictRequest(function_name="test_func", window=[0.1, 0.2, 0.3], index=1)
                response = stub.Predict(request)
                return response.result

        with futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda _: grpc_request(), range(num_requests)))

        # Ensure all results are non-negative
        for result in results:
            self.assertGreaterEqual(result, 0)

    def test_load_balancing(self):
        """Test if multiple gRPC processes share the workload."""
        num_requests = 50
        pid_results = set()

        def grpc_request():
            with grpc.insecure_channel(f"{HOST}:{PORT}") as channel:
                stub = pb2_grpc.ScalePredictorStub(channel)
                request = pb2.PredictRequest(function_name="test_func", window=[0.1, 0.2, 0.3], index=1)
                response, call = stub.Predict.with_call(request)

                pid = None
                for key, value in call.trailing_metadata():
                    if key == "pid":
                        pid = int(value)
                        pid_results.add(pid)

                return response.result

        with futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: grpc_request(), range(num_requests)))

        self.assertGreater(len(pid_results), 1, f"Expected multiple processes, but got: {pid_results}")

if __name__ == "__main__":
    unittest.main()
