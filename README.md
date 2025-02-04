# scale-predictor

## Description
A predictor to estimate how many instances to scale up based on historical invocation trace.

## Running the predictor

For local testing and development, run with the following command:
``` bash
pip install -r requirements.txt
python3 main.py
```
grpc server will be started on localhost:50051

## Run with Docker
You can directly pull and run scale-predictor from docker hub using the following command:
``` bash
docker pull zhaidea/scale-predictor:latest
docker run -p 50051:50051 zhaidea/scale-predictor
```

or you can build your own image by running Dockerfile:
``` bash
docker build -t scale-predictor .
docker run -p 50051:50051 scale-predictor
```
