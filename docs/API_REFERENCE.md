# JD-Agent API 参考文档

## 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `application/json`
- **API Docs**: `http://localhost:8000/docs` (Swagger UI)

---

## 接口列表

### 1. 健康检查

```
GET /api/v1/health
```

**Response:**
```json
{
    "status": "ok",
    "version": "1.0.0",
    "tools_count": 10
}
```

### 2. 聊天

```
POST /api/v1/chat
```

**Request:**
```json
{
    "message": "帮我查一下上个月的订单",
    "session_id": "abc123",
    "stream": false
}
```

**Response:**
```json
{
    "response": "查询结果：...",
    "session_id": "abc123"
}
```

### 3. 获取工具列表

```
GET /api/v1/tools
```

**Response:**
```json
{
    "tools": [
        {
            "name": "rag_query",
            "description": "RAG检索增强查询",
            "parameters": {"query": {"type": "string"}}
        }
    ]
}
```

### 4. 清空会话

```
POST /api/v1/session/{session_id}/clear
```

---

## 流式聊天（SSE）

暂未实现，后续将支持 Server-Sent Events 流式输出。
