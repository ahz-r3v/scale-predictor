import grpc
from concurrent import futures

from src.scale_predictor.server import ScalePredictorService
from src.scale_predictor.predictor import ScalePredictor

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc
from grpc_reflection.v1alpha import reflection

import os
import logging

if __name__ == "__main__":
    model_selector = os.getenv("PREDICTOR_MODEL", default='default')
    log_level_str = os.getenv("LOG_LEVEL", default="INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    cutoff_string = os.getenv("CUT_OFF", "0.02")
    try:
        cutoff_value = float(cutoff_string)
    except ValueError:
        print(f"Invalid CUT_OFF value: {cutoff_string}. Using default 0.02.")
        cutoff_value = 0.02

    logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
    logging.getLogger("lightning_fabric").setLevel(logging.CRITICAL)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("predictor.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("[VERSION] v0.1.6: Set outdated window buckets to -1...")
    if model_selector not in ["default", "linear", "historical", "nhits"]:
        logger.error(f"Invalid model selector: {model_selector}, using default model instead.")
        model_selector = "default"
    logger.info(f"Using model: {model_selector}")

    # Start grpc server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    grpc_service = ScalePredictorService(model_selector, cutoff_value)
    scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
        grpc_service,
        grpc_server
    )
    SERVICE_NAMES = (
        scale_predictor_pb2.DESCRIPTOR.services_by_name['ScalePredictor'].full_name,
        reflection.SERVICE_NAME,  # reflect service
    )
    reflection.enable_server_reflection(SERVICE_NAMES, grpc_server)

    grpc_server.add_insecure_port(f'[::]:{50051}')
    grpc_server.start()
    logger.info("gRPC server is running on port 50051...")
    
    grpc_server.wait_for_termination()

