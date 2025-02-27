import grpc
import logging
from concurrent import futures
import os

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc

from .predictor import ScalePredictor

class ScalePredictorService(scale_predictor_pb2_grpc.ScalePredictorServicer):
    def __init__(self, debug):
        self.predictor = ScalePredictor(debug)
        self.debug = debug
        self.logger = logging.getLogger(__name__)

    def Predict(self, request, context):
        function_name = request.function_name
        window = request.window
        index = request.index

        self.logger.debug(f"grpc 'Predict' received. function_name: {function_name}, window: {window}, index: {index}")
        
        try:
            result = self.predictor.predict(function_name, window, index)
        except KeyError as ke:
            self.logger.error(f'KeyError: {ke}, returns -1')
            context.set_details(str(ke))
            context.set_code(grpc.StatusCode.INTERNAL)
            return scale_predictor_pb2.PredictResponse(result=-1)

        self.logger.info(f'grpc returns: result: {result}')
        return scale_predictor_pb2.PredictResponse(
            result=result
        )

    def TrainByFile(self, request_iterator, context):
        filename = None
        window_size = None
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)

        with open("data/received.csv", "wb") as f:
            for chunk in request_iterator:
                if not filename:
                    filename = chunk.filename  # 记录文件名
                if not window_size:
                    window_size = chunk.window_size
                f.write(chunk.data)

        # check exist
        if not os.path.exists(filename):
            return scale_predictor_pb2.TrainStatus(success=False, message="File save failed!")

        # train
        result = self.predictor.train_by_file("./data/received.csv", window_size)
        if result == True:
            return scale_predictor_pb2.TrainStatus(success=True, message=f"Received {filename} and trained successfully!")

        return scale_predictor_pb2.TrainStatus(success=False, message=f"Received {filename} but Trained failed!")



def serve(debug, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scale_predictor_pb2_grpc.add_ScalePredictorServicer_to_server(
        ScalePredictorService(debug),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server listening on port {port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
