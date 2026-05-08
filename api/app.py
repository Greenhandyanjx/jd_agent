"""
FastAPI 主应用入口: API 服务
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="JD-Agent API",
    description="自实现 AI Agent 框架 API，支持 RAG、多工具调用、多轮对话",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


def run(host: str = "127.0.0.1", port: int = 8000):
    """启动 API 服务"""
    print(f"🚀 JD-Agent API 启动在 http://{host}:{port}")
    print(f"📄 API Docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
