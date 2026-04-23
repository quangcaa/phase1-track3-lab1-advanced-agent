"""Benchmark runner — chạy ReAct vs Reflexion trên HotpotQA.

Usage:
    # Mock mode (test nhanh):
    python run_benchmark.py --dataset data/hotpot_benchmark.json --out-dir outputs/mock_run --mode mock

    # Ollama mode (Qwen3.5:9b-Instruct):
    python run_benchmark.py --dataset data/hotpot_benchmark.json --out-dir outputs/final_run --mode ollama

    # OpenAI mode (GPT-4o-mini):
    python run_benchmark.py --dataset data/hotpot_benchmark.json --out-dir outputs/openai_run --mode openai
"""
from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_benchmark.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "mock",
) -> None:
    examples = load_dataset(dataset)
    print(f"[bold cyan]Mode:[/bold cyan] {mode}")
    print(f"[bold cyan]Dataset:[/bold cyan] {dataset} ({len(examples)} examples)")
    print(f"[bold cyan]Reflexion max attempts:[/bold cyan] {reflexion_attempts}")

    react = ReActAgent(mode=mode)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, mode=mode)

    react_records = []
    reflexion_records = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        # Chạy ReAct
        task1 = progress.add_task("[green]ReAct Agent...", total=len(examples))
        for example in examples:
            record = react.run(example)
            react_records.append(record)
            progress.advance(task1)

        # Chạy Reflexion
        task2 = progress.add_task("[blue]Reflexion Agent...", total=len(examples))
        for example in examples:
            record = reflexion.run(example)
            reflexion_records.append(record)
            progress.advance(task2)

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"\n[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(f"\n[bold]Summary:[/bold]")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
