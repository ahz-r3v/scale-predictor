import grpc
from concurrent import futures

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc

from .predictor import ScalePredictor

class ScalePredictorService(scale_predictor_pb2_grpc.ScalePredictorServicer):
    def __init__(self):
        self.predictor = ScalePredictor()

    def Predict(self, request, context):
        function_name = request.function_name
        window = request.window
        index = request.index
        
        try:
            result = self.predictor.predict(function_name, window, index)
        except RuntimeError as re:
            context.set_details(str(re))
            context.set_code(grpc.StatusCode.INTERNAL)
            return scale_predictor_pb2.PredictResponse(result=0)

        return scale_predictor_pb2.PredictResponse(
            result=result
        )

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
        ScalePredictorService(),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server listening on port {port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
