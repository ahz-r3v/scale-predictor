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

## Deploy with k8s
Upload your image:
``` bash
docker login -u <your_username> -p <yourpassword>
docker build -t <username>/scale-predictor-multiProcess:latest .
docker push <username>/scale-predictor-multiProcess:latest

```
Modify `zhaidea` in config/predictor.yaml to your username
On k8s clauster:
``` bash
kubectl apply -f config/predictor.yaml -n knative-serving
kubectl apply -f config/predictor-service.yaml -n knative-serving
```

## Set environment value  
On k8s cluster:  
```bash
# Use NHiTS model
kubectl set env deployment/scale-predictor -n knative-serving PREDICTOR_MODEL=nhits
# Use Linear Regressing
kubectl set env deployment/scale-predictor -n knative-serving PREDICTOR_MODEL=linear
# Use Historical
kubectl set env deployment/scale-predictor -n knative-serving PREDICTOR_MODEL=historical
# Use defalt window_average
kubectl set env deployment/scale-predictor -n knative-serving PREDICTOR_MODEL=default
```

