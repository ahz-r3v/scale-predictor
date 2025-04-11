import grpc
import logging
from concurrent import futures
import os
import tempfile
import uuid
import pandas as pd

import src.scale_predictor.scale_predictor_pb2 as scale_predictor_pb2
import src.scale_predictor.scale_predictor_pb2_grpc as scale_predictor_pb2_grpc

from .predictor import ScalePredictor

class ScalePredictorService(scale_predictor_pb2_grpc.ScalePredictorServicer):
    def __init__(self, model, cutoff_value):
        self.predictor = ScalePredictor(model, cutoff_value)
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)

    def Predict(self, request, context):
        function_name = request.function_name
        window = request.window
        index = request.index
        pid = os.getpid()

        self.logger.debug(f"[PID: {pid}] grpc 'Predict' received. function_name: {function_name}, window: {window}, index: {index}")
        
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
        """Handle file upload and model training request.
        
        Args:
            request_iterator: Iterator of FileChunk messages containing file data
            context: gRPC context
            
        Returns:
            TrainStatus message indicating success/failure
        """
        filename = None
        window_size = None
        
        # Create a temporary file with unique name
        temp_file = os.path.join(self.data_dir, f"upload_{uuid.uuid4()}.csv")
        self.logger.info(f"Starting file upload to {temp_file}")

        try:
            # Write chunks to temporary file
            with open(temp_file, "wb") as f:
                for chunk in request_iterator:
                    if not filename:
                        filename = chunk.filename
                    if not window_size:
                        window_size = chunk.window_size
                    if not chunk.data:
                        raise ValueError("Received empty data chunk")
                    f.write(chunk.data)

            if not filename or not window_size:
                raise ValueError("Missing required metadata (filename or window_size)")

            self.logger.info(f"File received: {filename}, window_size: {window_size}")
            
            # Validate CSV format
            try:
                df = pd.read_csv(temp_file)
                if df.empty:
                    raise ValueError("CSV file is empty")
            except Exception as e:
                raise ValueError(f"Invalid CSV format: {str(e)}")

            # generate dataframe
            train_set_filename = 'data/train.csv'
            test_set_filename = 'data/test.csv'
            self.generate_dataframe(temp_file, train_set_filename, test_set_filename)
            # Train model
            result = self.predictor.train_by_file(train_set_filename, window_size)
            if result:
                self.logger.info(f"Training successful for {filename}")
                return scale_predictor_pb2.TrainStatus(
                    success=True, 
                    message=f"File {filename} processed and model trained successfully"
                )
            else:
                raise RuntimeError("Model training failed")

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.logger.error(error_msg)
            return scale_predictor_pb2.TrainStatus(success=False, message=error_msg)
            
        finally:
            # Cleanup temporary file
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temporary file {temp_file}: {str(e)}")



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
