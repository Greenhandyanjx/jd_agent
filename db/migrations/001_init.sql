-- ============================================================
-- Migration 001: 初始化会话和消息表
-- ============================================================
--
-- 设计说明:
-- - 使用 sessions + messages 两层结构（符合数据库范式）
-- - sessions 存元数据，messages 存具体消息内容
-- - tool_calls 用 JSONB 存储（灵活应对不同 LLM 返回格式）
-- - metadata 用 JSONB 存储（扩展会话级别的自定义字段）
-- - 外键 CASCADE 删除：删会话时自动删其消息
-- - 时间戳统一用 TIMESTAMPTZ（带时区，避免时区问题）
-- ============================================================

-- ── 会话表 ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    key                TEXT PRIMARY KEY,         -- 会话唯一标识，如 "cli:direct" / "api:abc123"
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 创建时间
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 最后更新时间
    metadata           JSONB DEFAULT '{}',       -- 会话级扩展字段（JSON 对象）
    last_consolidated  INTEGER DEFAULT 0         -- 已 consolidation 的消息计数偏移
);

-- ── 消息表 ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id               BIGSERIAL PRIMARY KEY,      -- 自增主键，保证消息有序
    session_key      TEXT NOT NULL REFERENCES sessions(key) ON DELETE CASCADE,  -- 所属会话
    role             TEXT NOT NULL,               -- 角色: user / assistant / tool / system
    content          TEXT NOT NULL DEFAULT '',    -- 消息内容（空字符串而非 NULL，避免 LLM 渲染异常）
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 消息时间戳
    tool_calls       JSONB,                      -- assistant 消息的工具调用列表（JSON 数组）
    tool_call_id     TEXT,                       -- tool 消息关联的调用 ID
    name             TEXT                        -- tool 消息的工具名称
);

-- ── 索引 ──────────────────────────────────────────────

-- 按会话+时间排序查询（最常用的查询模式：加载一个会话的所有消息）
CREATE INDEX IF NOT EXISTS idx_messages_session_ts
    ON messages(session_key, timestamp, id);

-- 按角色过滤（用于分析统计）
CREATE INDEX IF NOT EXISTS idx_messages_role
    ON messages(role);

-- ── 自动更新 updated_at 的触发器 ───────────────────────

-- 说明：当 messages 表有变更时，自动更新对应会话的 updated_at
-- 这样就不需要业务代码手动维护这个字段

CREATE OR REPLACE FUNCTION touch_session_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE sessions
    SET updated_at = NOW()
    WHERE key = COALESCE(NEW.session_key, OLD.session_key);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 删除已存在的触发器（避免重复创建报错）
DROP TRIGGER IF EXISTS trg_messages_touch_session ON messages;

-- INSERT / UPDATE / DELETE 消息时都触发
CREATE TRIGGER trg_messages_touch_session
    AFTER INSERT OR UPDATE OR DELETE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION touch_session_updated_at();
