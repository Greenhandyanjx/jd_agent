"""
RAG 召回率评估工具
==================
测量 RAG 检索系统的 recall@k、precision@k、MRR、F1 等指标。

用法：
    # 使用内置测试集进行评估
    python scripts/eval_rag.py

    # 使用自定义测试集
    python scripts/eval_rag.py --test-set data/eval/my_test_set.json

    # 指定 top-k
    python scripts/eval_rag.py --k 3 5 10

    # 输出详细报告到文件
    python scripts/eval_rag.py --output eval_report.md

指标说明：
    - recall@k:    前 k 个结果中，命中的相关文档数 / 总相关文档数
    - precision@k: 前 k 个结果中，命中的相关文档数 / k
    - MRR:         第一个相关文档出现在结果中的位置的倒数均值
    - F1@k:        precision@k 和 recall@k 的调和平均数
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── 确保能找到项目根目录 ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────
# 内置测试集
# ─────────────────────────────────────────────────────────

BUILTIN_TEST_SET = [
    # ─── 选购指南类 ───
    {
        "query": "小户型适合什么样的扫地机器人？",
        "expected_keywords": ["轻便灵活", "机身高度", "续航"],
        "source_hint": "选购指南",
    },
    {
        "query": "扫地机器人吸力多大够用？",
        "expected_keywords": ["3000Pa", "4000Pa", "吸力"],
        "source_hint": "选购指南",
    },
    {
        "query": "扫地机器人哪种导航技术好？",
        "expected_keywords": ["激光导航", "视觉导航", "导航技术"],
        "source_hint": "选购指南",
    },
    {
        "query": "宠物家庭怎么选扫地机器人？",
        "expected_keywords": ["宠物", "防缠绕", "毛发"],
        "source_hint": "选购指南",
    },
    {
        "query": "扫地机器人自动洗拖布功能重要吗？",
        "expected_keywords": ["自动洗拖布", "热风烘干", "拖布"],
        "source_hint": "选购指南",
    },
    {
        "query": "扫地机器人续航多久合适？",
        "expected_keywords": ["续航", "电池", "充电"],
        "source_hint": "选购指南",
    },
    {
        "query": "扫地机器人避障能力怎么看？",
        "expected_keywords": ["避障", "3D结构光", "传感器"],
        "source_hint": "选购指南",
    },
    {
        "query": "木地板适合什么样的扫地机器人？",
        "expected_keywords": ["木地板", "拖布可抬升", "出水量"],
        "source_hint": "选购指南",
    },
    {
        "query": "扫地机器人有哪些品牌推荐？",
        "expected_keywords": ["科沃斯", "石头", "品牌"],
        "source_hint": "选购指南",
    },
    {
        "query": "扫地机器人集尘功能有必要吗？",
        "expected_keywords": ["集尘", "集尘袋", "自动集尘"],
        "source_hint": "选购指南",
    },

    # ─── 故障排除类 ───
    {
        "query": "扫地机器人开机没反应怎么办？",
        "expected_keywords": ["开机无反应", "电源", "电池"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人无法连接WiFi怎么解决？",
        "expected_keywords": ["WiFi", "网络", "连接"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人吸力变小了是什么原因？",
        "expected_keywords": ["吸力", "尘盒", "滤网", "堵塞"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人找不到充电座怎么办？",
        "expected_keywords": ["找不到充电座", "回充", "传感器"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人拖地不出水怎么处理？",
        "expected_keywords": ["拖地不出水", "出水管", "水箱"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人电池续航衰减严重怎么办？",
        "expected_keywords": ["电池续航", "衰减", "充电"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人主刷不转了怎么修？",
        "expected_keywords": ["主刷", "不旋转", "电机"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人频繁碰撞家具怎么回事？",
        "expected_keywords": ["碰撞", "避障", "传感器"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人漏水怎么办？",
        "expected_keywords": ["漏水", "水箱", "密封圈"],
        "source_hint": "故障排除",
    },
    {
        "query": "扫地机器人噪音突然变大是什么问题？",
        "expected_keywords": ["噪音", "主刷", "缠绕"],
        "source_hint": "故障排除",
    },

    # ─── 100问类 ───
    {
        "query": "首次使用扫地机器人需要做什么？",
        "expected_keywords": ["首次使用", "建图", "充电"],
        "source_hint": "扫地机器人100问",
    },
    {
        "query": "扫地机器人建图不完整怎么办？",
        "expected_keywords": ["建图", "地图", "反光物"],
        "source_hint": "扫地机器人100问",
    },
    {
        "query": "如何设置扫地机器人定时清扫？",
        "expected_keywords": ["定时清扫", "APP", "设置"],
        "source_hint": "扫地机器人100问",
    },
]


# ─────────────────────────────────────────────────────────
# 评估指标计算
# ─────────────────────────────────────────────────────────

def keyword_match_score(text: str, keywords: list[str]) -> float:
    """
    检查文本中是否包含期望的关键词。
    返回匹配率 (0.0~1.0)。
    """
    if not keywords:
        return 1.0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


def compute_metrics(
    results: list[dict[str, Any]],
    k_values: list[int],
) -> dict[str, Any]:
    """
    计算各项评估指标。

    Args:
        results: 每条测试用例的检索结果
        k_values: 要评估的 top-k 值列表

    Returns:
        包含各项指标的字典
    """
    n = len(results)
    if n == 0:
        return {}

    # 汇总所有 k 值的指标
    metrics: dict[str, Any] = {
        "total_queries": n,
        "k_values": k_values,
    }

    # 每个 k 的指标
    for k in k_values:
        recall_list = []
        precision_list = []
        f1_list = []
        hit_count = 0  # 至少命中一个相关文档的查询数

        for r in results:
            retrieved = r["retrieved"][:k]
            if not retrieved:
                recall_list.append(0.0)
                precision_list.append(0.0)
                f1_list.append(0.0)
                continue

            # 取每条 retrieved 文档对 query 的关键词匹配分作为"相关性"信号
            keyword_hits = [d["keyword_score"] for d in retrieved]
            # 如果某条文档的关键词匹配率 >= 阈值，视为命中
            relevant_found = sum(1 for s in keyword_hits if s >= 0.3)

            # Recall@k: 至少命中一个相关文档即为 1，否则为 0
            recall = 1.0 if relevant_found > 0 else 0.0
            precision = relevant_found / k
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)

            if relevant_found > 0:
                hit_count += 1

        metrics[f"recall@{k}"] = round(sum(recall_list) / n, 4)
        metrics[f"precision@{k}"] = round(sum(precision_list) / n, 4)
        metrics[f"f1@{k}"] = round(sum(f1_list) / n, 4)
        metrics[f"hit_rate@{k}"] = round(hit_count / n, 4)

    # ── MRR (Mean Reciprocal Rank) ──
    mrr_sum = 0.0
    for r in results:
        rank = 0
        for i, d in enumerate(r["retrieved"], 1):
            if d["keyword_score"] >= 0.3:
                rank = i
                break
        mrr_sum += 1.0 / rank if rank > 0 else 0.0
    metrics["mrr"] = round(mrr_sum / n, 4)

    # ── 平均检索延迟 ──
    latencies = [r.get("latency_ms", 0) for r in results]
    metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else 0
    metrics["max_latency_ms"] = round(max(latencies), 1) if latencies else 0

    return metrics


# ─────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────

def format_metrics_table(metrics: dict[str, Any]) -> str:
    """生成 Markdown 格式的指标表格"""
    k_values = metrics.get("k_values", [3, 5, 10])

    lines = ["### 检索召回率评估报告", "", "| 指标 |", ""]

    # 按 k 分组的指标
    header = "| k | Recall@k | Precision@k | F1@k | HitRate@k |"
    sep = "|---|" + "---|" * 4
    lines.append(header)
    lines.append(sep)

    for k in k_values:
        recall = metrics.get(f"recall@{k}", "-")
        precision = metrics.get(f"precision@{k}", "-")
        f1 = metrics.get(f"f1@{k}", "-")
        hit = metrics.get(f"hit_rate@{k}", "-")
        lines.append(f"| {k} | {recall} | {precision} | {f1} | {hit} |")

    lines.append("")
    lines.append(f"- **MRR (Mean Reciprocal Rank)**: {metrics.get('mrr', '-')}")
    lines.append(f"- **总查询数**: {metrics.get('total_queries', 0)}")
    lines.append(f"- **平均检索延迟**: {metrics.get('avg_latency_ms', '-')} ms")
    lines.append(f"- **最大检索延迟**: {metrics.get('max_latency_ms', '-')} ms")

    return "\n".join(lines)


def format_detail_report(
    test_set: list[dict],
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> str:
    """生成详细的逐条评估报告"""
    lines = ["# RAG 召回率评估报告", "", f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    # 汇总表
    lines.append(format_metrics_table(metrics))
    lines.append("")

    # 逐条详情
    lines.append("---")
    lines.append("### 逐条查询详情")
    lines.append("")

    for i, (item, r) in enumerate(zip(test_set, results), 1):
        query = item["query"]
        kw = item.get("expected_keywords", [])
        hint = item.get("source_hint", "")
        latency = r.get("latency_ms", 0)
        retrieved = r["retrieved"]

        lines.append(f"#### {i}. {query}")
        lines.append(f"- **期望关键词**: `{'`, `'.join(kw)}`")
        lines.append(f"- **来源**: {hint}")
        lines.append(f"- **检索耗时**: {latency} ms")
        lines.append(f"- **召回文档数**: {len(retrieved)}")
        lines.append("")

        if retrieved:
            lines.append("| # | 关键词匹配 | 片段预览 |")
            lines.append("|---|----------|---------|")
            for j, d in enumerate(retrieved, 1):
                score = d["keyword_score"]
                preview = d["content"][:80].replace("\n", " ")
                lines.append(f"| {j} | {score:.2f} | {preview} |")
        else:
            lines.append("*(未检索到任何文档)*")

        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 主评估逻辑
# ─────────────────────────────────────────────────────────

def run_evaluation(
    test_set: list[dict],
    k_values: list[int] = None,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    运行 RAG 召回率评估。

    Args:
        test_set: 测试集，每项包含 query 和 expected_keywords
        k_values: 评估的 top-k 值列表，默认 [3, 5, 10]
        verbose: 是否打印详细日志

    Returns:
        (逐条结果明细, 汇总指标)
    """
    if k_values is None:
        k_values = [3, 5, 10]

    from rag.rag_service import RagRetrievalService

    rag = RagRetrievalService(enable_optimization=True)
    max_k = max(k_values)

    results = []

    for i, item in enumerate(test_set):
        query = item["query"]
        keywords = item.get("expected_keywords", [])

        if verbose:
            print(f"  [{i+1}/{len(test_set)}] 查询: {query}")

        # 计时
        t0 = time.time()

        # 通过 RAG 检索（走完整管线：Rewrite → Hybrid → Reranker）
        # 注意：要拿原始的 Document 对象而不是格式化文本
        # 因为我们需要对每个片段计算关键词匹配分
        docs, _ = rag.retrieve_with_reranker(query, k=max_k)

        latency = (time.time() - t0) * 1000  # ms

        # 对检索结果打分
        retrieved = []
        for doc in docs:
            score = keyword_match_score(doc.page_content, keywords)
            retrieved.append({
                "content": doc.page_content,
                "score": doc.metadata.get("score", 0),
                "rrf_score": doc.metadata.get("rrf_score", 0),
                "vector_rank": doc.metadata.get("vector_rank", 0),
                "bm25_rank": doc.metadata.get("bm25_rank", 0),
                "keyword_score": score,
            })

        if verbose:
            matched = sum(1 for d in retrieved if d["keyword_score"] >= 0.3)
            print(f"    → {len(retrieved)} 篇, {matched} 篇命中关键词, {latency:.0f}ms")

        results.append({
            "query": query,
            "expected_keywords": keywords,
            "retrieved": retrieved,
            "latency_ms": round(latency, 1),
        })

    # 计算汇总指标
    metrics = compute_metrics(results, k_values)

    return results, metrics


# ─────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG 召回率评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        help="测试集 JSON 文件路径（默认使用内置测试集）",
        default=None,
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="评估的 top-k 值列表（默认 3 5 10）",
    )
    parser.add_argument(
        "--output",
        help="输出报告文件路径（默认打印到终端）",
        default=None,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示逐条查询的详细结果",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="输出逐条查询的详细报告（含每个召回片段的匹配分）",
    )

    args = parser.parse_args()

    # ── 加载测试集 ──
    if args.test_set:
        test_set_path = Path(args.test_set)
        if not test_set_path.exists():
            print(f"❌ 测试集文件不存在: {test_set_path}")
            sys.exit(1)
        with open(test_set_path, "r", encoding="utf-8") as f:
            test_set = json.load(f)
        print(f"✅ 加载自定义测试集: {test_set_path} ({len(test_set)} 条)")
    else:
        test_set = BUILTIN_TEST_SET
        print(f"✅ 使用内置测试集 ({len(test_set)} 条)")

    # ── 运行评估 ──
    print(f"\n🔍 开始 RAG 召回率评估...")
    print(f"   评估 k 值: {args.k}")
    print()

    t_start = time.time()
    results, metrics = run_evaluation(
        test_set,
        k_values=args.k,
        verbose=args.verbose,
    )
    elapsed = time.time() - t_start

    print(f"\n✅ 评估完成 ({elapsed:.1f}s)\n")

    # ── 输出报告 ──
    if args.detail:
        report = format_detail_report(test_set, results, metrics)
    else:
        report = format_metrics_table(metrics)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"📄 报告已保存: {output_path}")
    else:
        print(report)

    # ── 简要结论 ──
    print()
    best_k = max(args.k)
    print("📊 小结:")
    print(f"   查询总数: {metrics['total_queries']}")
    print(f"   MRR:      {metrics['mrr']}")
    for k in args.k:
        print(f"   Recall@{k}:  {metrics.get(f'recall@{k}', '-')}")
        print(f"   F1@{k}:     {metrics.get(f'f1@{k}', '-')}")
    print(f"   平均延迟:  {metrics.get('avg_latency_ms', '-')} ms")


if __name__ == "__main__":
    main()
