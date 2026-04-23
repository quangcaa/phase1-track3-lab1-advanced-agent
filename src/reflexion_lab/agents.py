"""Reflexion Agent — ReAct và Reflexion Agent với runtime toggle.

Hỗ trợ 3 mode:
- mock: dùng mock_runtime.py (test nhanh, không cần LLM)
- ollama: dùng llm_runtime.py (Qwen3.5-9B-Instruct qua Ollama)
- openai: dùng openai_runtime.py (GPT-4o-mini qua OpenAI API)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

# Import cả 2 runtime — chọn theo mode
from . import mock_runtime
from . import llm_runtime
from . import openai_runtime


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    mode: str = "mock"  # "mock", "ollama", hoặc "openai"

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            if self.mode == "ollama":
                # --- Ollama LLM Runtime ---
                answer, actor_tokens, actor_latency = llm_runtime.actor_answer(
                    example, attempt_id, self.agent_type, reflection_memory
                )
                judge, eval_tokens, eval_latency = llm_runtime.evaluator(example, answer)
                token_estimate = actor_tokens + eval_tokens
                latency_ms = actor_latency + eval_latency
            elif self.mode == "openai":
                # --- OpenAI GPT-4o-mini Runtime ---
                answer, actor_tokens, actor_latency = openai_runtime.actor_answer(
                    example, attempt_id, self.agent_type, reflection_memory
                )
                judge, eval_tokens, eval_latency = openai_runtime.evaluator(example, answer)
                token_estimate = actor_tokens + eval_tokens
                latency_ms = actor_latency + eval_latency
            else:
                # --- Mock Runtime ---
                answer = mock_runtime.actor_answer(
                    example, attempt_id, self.agent_type, reflection_memory
                )
                judge = mock_runtime.evaluator(example, answer)
                # Mock: ước tính token và latency
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            # --- Reflexion loop ---
            # Khi agent_type là "reflexion" và chưa hết số lần attempt:
            # 1. Gọi reflector để phân tích lỗi
            # 2. Cập nhật reflection_memory để Actor dùng cho lần sau
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                if self.mode in ("ollama", "openai"):
                    runtime = llm_runtime if self.mode == "ollama" else openai_runtime
                    reflection, ref_tokens, ref_latency = runtime.reflector(
                        example, attempt_id, judge
                    )
                    token_estimate += ref_tokens
                    latency_ms += ref_latency
                    # Cập nhật trace với token/latency bao gồm reflection
                    trace.token_estimate = token_estimate
                    trace.latency_ms = latency_ms
                else:
                    reflection = mock_runtime.reflector(example, attempt_id, judge)

                # Thêm bài học vào reflection_memory (sliding window: giữ 3 gần nhất)
                memory_entry = f"Attempt {attempt_id}: {reflection.lesson} Strategy: {reflection.next_strategy}"
                reflection_memory.append(memory_entry)
                if len(reflection_memory) > 3:
                    reflection_memory = reflection_memory[-3:]

                # Ghi nhận reflection
                trace.reflection = reflection
                reflections.append(reflection)

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)

        # Phân loại failure mode
        if final_score == 1:
            failure_mode = "none"
        elif self.mode == "ollama":
            failure_mode = llm_runtime._classify_failure_mode(
                traces[-1].reason if traces else ""
            )
        elif self.mode == "openai":
            failure_mode = openai_runtime._classify_failure_mode(
                traces[-1].reason if traces else ""
            )
        else:
            failure_mode = mock_runtime.FAILURE_MODE_BY_QID.get(
                example.qid, "wrong_final_answer"
            )

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, mode: str = "mock") -> None:
        super().__init__(agent_type="react", max_attempts=1, mode=mode)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, mode: str = "mock") -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, mode=mode)
