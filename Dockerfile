# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 环境变量
ENV PYTHONPATH=/app
ENV DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}

# 暴露端口
EXPOSE 8000
EXPOSE 8501

# 默认启动 FastAPI
CMD ["python", "-m", "api.app"]
