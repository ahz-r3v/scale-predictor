# 使用轻量级的 Python 镜像
FROM python:3.11

# 设置工作目录
WORKDIR /scale-predictor

# 复制依赖文件，并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件到镜像中
COPY . .

# 暴露 gRPC 服务端口
EXPOSE 50051

# 运行服务
CMD ["python", "main.py"]
