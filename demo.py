import grpc
from concurrent import futures

from src.scale_predictor.server import ScalePredictorService
from src.scale_predictor.predictor import ScalePredictor

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc

if __name__ == "__main__":
    dataset = {
        "funcA": [10, 12, 15, 18, 20, 26, 30, 28, 23, 0, 23, 54, 34, 4, 435, 54, 0, 0, 0, 34, 4, 3, 2, 5, 0, 1],
        "funcB": [2, 3, 5, 7, 9, 9, 10, 5, 34, 23, 5, 1, 1, 0]
    }
    predictor = ScalePredictor()

    # Training.
    predictor.train(dataset, window_size=5)

    current_window_a = [35, 32, 31, 40, 50, 36, 3]
    needed_a = predictor.predict("funcA", current_window_a, 6)
    print(f"Predicted needed instances for funcA = {needed_a}")

    current_window_b = [9, 12, 14, 8, 11, 99]
    needed_b = predictor.predict("funcB", current_window_b, 1)
    print(f"Predicted needed instances for funcB = {needed_b}")

    # Start grpc server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpc_service = ScalePredictorService()
    grpc_service.predictor = predictor
    scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
        grpc_service,
        grpc_server
    )
    grpc_server.add_insecure_port(f'[::]:{50051}')

    grpc_server.start()

    grpc_server.wait_for_termination()

