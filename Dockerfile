FROM python:3.11

WORKDIR /scale-predictor

COPY requirements_darts.txt .
RUN pip install --no-cache-dir -r requirements_darts.txt

COPY . .

EXPOSE 50051

CMD ["python", "main.py"]
