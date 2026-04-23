"""Download HotpotQA benchmark data từ HuggingFace.

Tải 120 câu hỏi từ dataset hotpot_qa (config: distractor, split: train)
- 40 câu easy
- 40 câu medium
- 40 câu hard

Output: data/hotpot_benchmark.json
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from datasets import load_dataset


def download_hotpotqa(
    config: str = "distractor",
    split: str = "train",
    num_easy: int = 40,
    num_medium: int = 40,
    num_hard: int = 40,
    seed: int = 42,
    output_path: str = "data/hotpot_benchmark.json",
) -> None:
    print(f"Loading HotpotQA dataset (config={config}, split={split})...")
    ds = load_dataset("hotpot_qa", config, split=split)
    print(f"Total examples in {split}: {len(ds)}")

    # Phân nhóm theo level (easy/medium/hard)
    by_level: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for row in ds:
        level = row.get("level", "").lower()
        if level in by_level:
            by_level[level].append(row)

    for level, items in by_level.items():
        print(f"  {level}: {len(items)} examples available")

    # Shuffle và lấy đúng số lượng
    rng = random.Random(seed)
    counts = {"easy": num_easy, "medium": num_medium, "hard": num_hard}
    selected: list[dict] = []

    for level, count in counts.items():
        pool = by_level[level]
        if len(pool) < count:
            print(f"  WARNING: only {len(pool)} {level} examples available, requested {count}")
            count = len(pool)
        rng.shuffle(pool)
        selected.extend(pool[:count])

    # Shuffle lại toàn bộ để trộn đều
    rng.shuffle(selected)

    # Convert sang format QAExample
    results = []
    for idx, row in enumerate(selected):
        # HotpotQA format: context = [[titles...], [sentences_lists...]]
        # context[0] = list of paragraph titles
        # context[1] = list of sentence lists (mỗi paragraph là list[str])
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]

        context_chunks = []
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            context_chunks.append({
                "title": title,
                "text": text,
            })

        level = row["level"].lower()
        results.append({
            "qid": row["id"],
            "difficulty": level,
            "question": row["question"],
            "gold_answer": row["answer"],
            "context": context_chunks,
        })

    # Lưu file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Thống kê
    difficulty_counts = {}
    for r in results:
        d = r["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    print(f"\nSaved {len(results)} examples to {out}")
    print(f"Distribution: {difficulty_counts}")


if __name__ == "__main__":
    download_hotpotqa()
