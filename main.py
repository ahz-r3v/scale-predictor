import grpc
from concurrent import futures

from src.scale_predictor.server import ScalePredictorService
from src.scale_predictor.predictor import ScalePredictor

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc

import os

if __name__ == "__main__":
    debug = os.getenv("PD_DEBUG", default='0')
    predictor = ScalePredictor(debug)

    # Start grpc server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpc_service = ScalePredictorService(debug)
    grpc_service.predictor = predictor
    scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
        grpc_service,
        grpc_server
    )
    grpc_server.add_insecure_port(f'[::]:{50051}')

    grpc_server.start()

    grpc_server.wait_for_termination()

