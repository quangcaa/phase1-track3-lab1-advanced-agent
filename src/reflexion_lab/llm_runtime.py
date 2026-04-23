"""LLM Runtime — Gọi Qwen2.5-7B-Instruct qua Ollama local.

Thay thế mock_runtime.py khi chạy benchmark thật.
- actor_answer: gọi LLM để trả lời câu hỏi multi-hop
- evaluator: gọi LLM để chấm điểm structured JSON
- reflector: gọi LLM để rút bài học từ lỗi
- Token counting thật từ Ollama response
- Latency đo bằng time.perf_counter()
"""
from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
from typing import Any

from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .utils import normalize_answer

MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Failure mode mapping dựa trên phân tích lỗi
FAILURE_MODES = {
    "incomplete_multi_hop": ["first hop", "intermediate", "partial", "one hop", "stopped"],
    "entity_drift": ["wrong entity", "different", "confused", "mixed up", "drift"],
    "wrong_final_answer": ["incorrect", "wrong", "not match"],
    "looping": ["same answer", "repeated", "loop"],
    "reflection_overfit": ["overfit", "overcorrect"],
}


def _ollama_chat(messages: list[dict], json_format: bool = False) -> dict[str, Any]:
    """Gọi Ollama API chat endpoint.
    
    Returns dict với keys: message.content, prompt_eval_count, eval_count, total_duration
    """
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }
    if json_format:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Không thể kết nối Ollama tại {OLLAMA_URL}. "
            f"Hãy chắc chắn Ollama đang chạy (`ollama serve`). Error: {e}"
        )


def _classify_failure_mode(reason: str) -> str:
    """Phân loại failure mode từ reason của evaluator."""
    reason_lower = reason.lower()
    for mode, keywords in FAILURE_MODES.items():
        if any(kw in reason_lower for kw in keywords):
            return mode
    return "wrong_final_answer"


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int]:
    """Gọi Actor LLM để trả lời câu hỏi.
    
    Returns: (answer, token_count, latency_ms)
    """
    # Xây dựng context string từ các đoạn
    context_str = "\n\n".join(
        f"### {chunk.title}\n{chunk.text}" for chunk in example.context
    )

    # Xây dựng user prompt
    user_parts = [
        f"Context:\n{context_str}",
        f"\nQuestion: {example.question}",
    ]

    # Thêm reflection memory nếu có (chỉ cho Reflexion agent)
    if reflection_memory and agent_type == "reflexion":
        memory_str = "\n".join(f"- {m}" for m in reflection_memory[-3:])  # Sliding window: giữ 3 bài học gần nhất
        user_parts.append(f"\nReflection notes from previous attempts:\n{memory_str}")
        user_parts.append(f"\nThis is attempt #{attempt_id}. Use the reflection notes to improve your answer.")

    user_msg = "\n".join(user_parts)

    messages = [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    start = time.perf_counter()
    response = _ollama_chat(messages, json_format=False)
    latency_ms = int((time.perf_counter() - start) * 1000)

    answer = response.get("message", {}).get("content", "").strip()
    # Lấy token count thật từ Ollama
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens

    return answer, total_tokens, latency_ms


def evaluator(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, int]:
    """Gọi Evaluator LLM để chấm điểm.
    
    Returns: (JudgeResult, token_count, latency_ms)
    """
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n\n"
        f"Evaluate the predicted answer against the gold answer."
    )

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    start = time.perf_counter()
    response = _ollama_chat(messages, json_format=True)
    latency_ms = int((time.perf_counter() - start) * 1000)

    content = response.get("message", {}).get("content", "{}")
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens

    try:
        result_dict = json.loads(content)
        judge = JudgeResult(
            score=1 if result_dict.get("score", 0) == 1 else 0,
            reason=result_dict.get("reason", "No reason provided"),
            missing_evidence=result_dict.get("missing_evidence", []),
            spurious_claims=result_dict.get("spurious_claims", []),
        )
    except (json.JSONDecodeError, Exception):
        # Fallback: so sánh trực tiếp nếu LLM trả output lỗi
        is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        judge = JudgeResult(
            score=1 if is_correct else 0,
            reason="Fallback: direct string comparison" if is_correct else "Fallback: answers do not match",
        )

    return judge, total_tokens, latency_ms


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, int]:
    """Gọi Reflector LLM để rút bài học.
    
    Returns: (ReflectionEntry, token_count, latency_ms)
    """
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Attempt #{attempt_id} evaluation:\n"
        f"- Score: {judge.score}\n"
        f"- Reason: {judge.reason}\n"
        f"- Missing evidence: {judge.missing_evidence}\n"
        f"- Spurious claims: {judge.spurious_claims}\n\n"
        f"Analyze this failure and provide reflection for attempt #{attempt_id}."
    )

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    start = time.perf_counter()
    response = _ollama_chat(messages, json_format=True)
    latency_ms = int((time.perf_counter() - start) * 1000)

    content = response.get("message", {}).get("content", "{}")
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens

    try:
        result_dict = json.loads(content)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=result_dict.get("failure_reason", judge.reason),
            lesson=result_dict.get("lesson", "Review all evidence before answering."),
            next_strategy=result_dict.get("next_strategy", "Carefully trace each hop of reasoning."),
        )
    except (json.JSONDecodeError, Exception):
        # Fallback nếu LLM trả output lỗi
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Review all context paragraphs and complete all reasoning hops.",
            next_strategy="Trace the chain of evidence step by step before answering.",
        )

    return entry, total_tokens, latency_ms
