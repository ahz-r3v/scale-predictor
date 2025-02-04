import grpc
from concurrent import futures

from src.scale_predictor.server import ScalePredictorService
from src.scale_predictor.predictor import ScalePredictor

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc

import os
import logging

if __name__ == "__main__":
    debug = os.getenv("PD_DEBUG", default='0')

    logging.basicConfig(
    filename="scale-predictor.log", 
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

    # Start grpc server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpc_service = ScalePredictorService(debug)
    scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
        grpc_service,
        grpc_server
    )
    grpc_server.add_insecure_port(f'[::]:{50051}')
    grpc_server.start()
    logging.info("grpc_client started.")
    

    grpc_server.wait_for_termination()

