# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_benchmark.json
- Mode: openai
- Records: 240
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.85 | 0.9167 | 0.0667 |
| Avg attempts | 1 | 1.2417 | 0.2417 |
| Avg token estimate | 1827.03 | 2374.59 | 547.56 |
| Avg latency (ms) | 2769.72 | 3700.68 | 930.96 |

## Failure modes
```json
{
  "react": {
    "none": 102,
    "wrong_final_answer": 17,
    "entity_drift": 1
  },
  "reflexion": {
    "none": 110,
    "wrong_final_answer": 9,
    "entity_drift": 1
  },
  "combined": {
    "none": 212,
    "wrong_final_answer": 26,
    "entity_drift": 2
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Benchmark chạy trên 240 records (dataset: hotpot_benchmark.json, mode: openai). ReAct đạt EM = 0.85, Reflexion đạt EM = 0.9167 (delta = +0.0667). Reflexion sử dụng trung bình 1.2 attempts/câu so với 1.0 của ReAct, tốn thêm ~548 tokens và ~931ms latency. Reflexion giúp sửa lỗi khi attempt đầu dừng ở hop 1 hoặc drift sang entity sai. Tradeoff là chi phí token và latency cao hơn. Reflection memory (sliding window 3 entries) giữ bài học ngắn gọn, actionable, tránh context bloat. Structured evaluator (JudgeResult với score/reason/missing_evidence/spurious_claims) giúp reflector sinh bài học cụ thể hơn so với free-form evaluation.
