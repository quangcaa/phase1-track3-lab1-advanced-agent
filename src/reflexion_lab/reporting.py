from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    combined: Counter = Counter()
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1
        combined[record.failure_mode] += 1
    result = {agent: dict(counter) for agent, counter in grouped.items()}
    result["combined"] = dict(combined)
    return result

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    
    summary = summarize(records)
    failures = failure_breakdown(records)
    
    # Extensions đã triển khai
    extensions = [
        "structured_evaluator",       # Evaluator trả JSON structured (JudgeResult)
        "reflection_memory",          # Reflection memory truyền qua attempts (sliding window)
        "benchmark_report_json",      # Xuất report.json đầy đủ format
        "mock_mode_for_autograding",  # Giữ mock mode hoạt động song song
    ]
    
    # Sinh discussion động từ kết quả thực tế
    react_stats = summary.get("react", {})
    reflexion_stats = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})
    
    discussion_parts = []
    discussion_parts.append(
        f"Benchmark chạy trên {len(records)} records (dataset: {dataset_name}, mode: {mode})."
    )
    
    if react_stats and reflexion_stats:
        discussion_parts.append(
            f"ReAct đạt EM = {react_stats.get('em', 0)}, "
            f"Reflexion đạt EM = {reflexion_stats.get('em', 0)} "
            f"(delta = {delta.get('em_abs', 0):+.4f})."
        )
        discussion_parts.append(
            f"Reflexion sử dụng trung bình {reflexion_stats.get('avg_attempts', 0):.1f} attempts/câu "
            f"so với {react_stats.get('avg_attempts', 0):.1f} của ReAct, "
            f"tốn thêm ~{delta.get('tokens_abs', 0):.0f} tokens và ~{delta.get('latency_abs', 0):.0f}ms latency."
        )
    
    discussion_parts.append(
        "Reflexion giúp sửa lỗi khi attempt đầu dừng ở hop 1 hoặc drift sang entity sai. "
        "Tradeoff là chi phí token và latency cao hơn. "
        "Reflection memory (sliding window 3 entries) giữ bài học ngắn gọn, actionable, tránh context bloat. "
        "Structured evaluator (JudgeResult với score/reason/missing_evidence/spurious_claims) giúp reflector "
        "sinh bài học cụ thể hơn so với free-form evaluation."
    )
    
    discussion = " ".join(discussion_parts)
    
    return ReportPayload(
        meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})},
        summary=summary,
        failure_modes=failures,
        examples=examples,
        extensions=extensions,
        discussion=discussion,
    )

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
