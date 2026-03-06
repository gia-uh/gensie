import typer
import uvicorn
import httpx
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.progress import track
from collections import defaultdict
from gensie.task import Task
from gensie.eval import Evaluator, flatten_json

app = typer.Typer(help="GenSIE Developer Tools")
console = Console()


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Starts the FastAPI server for the agent."""
    console.print(f"[bold green]Starting GenSIE Agent Server on {host}:{port}...[/bold green]")
    uvicorn.run("gensie.server:app", host=host, port=port, reload=True)


@app.command()
def eval(
    data: Path = typer.Option(..., help="Path to directory containing JSON tasks"),
    url: str = typer.Option("http://localhost:8000", help="Agent service URL"),
    pipeline: str = typer.Option("baseline", help="Name of the pipeline to evaluate"),
    model: str = typer.Option(..., help="The exact model name to use for inference"),
    limit: Optional[int] = typer.Option(None, help="Limit the number of tasks to evaluate"),
    output: Optional[Path] = typer.Option(None, help="Path to save the JSON evaluation report"),
):
    """Evaluates the agent against a local dataset and generates a report."""
    evaluator = Evaluator()

    if not data.is_dir():
        console.print(f"[bold red]Error: {data} is not a directory.[/bold red]")
        raise typer.Exit(1)

    json_files = list(data.rglob("*.json"))
    if limit:
        json_files = json_files[:limit]

    tps_list = []
    gold_counts = []
    system_counts = []
    individual_results: List[Dict[str, Any]] = []

    results_table = Table(title=f"Evaluation Results (Pipeline: {pipeline})")
    results_table.add_column("Task ID", style="cyan")
    results_table.add_column("TPS", justify="right")
    results_table.add_column("Gold Keys", justify="right")
    results_table.add_column("System Keys", justify="right")
    results_table.add_column("Status", justify="center")

    params = {"pipeline": pipeline}
    if model:
        params["model"] = model

    participant_info = {"team_name": "Unknown", "institution": "Unknown"}

    with httpx.Client(timeout=60.0) as client:
        # 1. Fetch Participant Info
        try:
            info_resp = client.get(f"{url}/info")
            info_resp.raise_for_status()
            participant_info = info_resp.json()
            console.print(
                f"Auditing Team: [bold magenta]{participant_info.get('team_name', 'Unknown')}[/bold magenta]"
            )
        except Exception as e:
            console.print(f"[yellow]Could not retrieve participant info: {e}[/yellow]")

        # 2. Process Tasks
        for file_path in track(json_files, description="Processing tasks..."):
            task = None
            try:
                task = Task.load(file_path)
                # Call agent
                resp = client.post(
                    f"{url}/run", json=task.model_dump(mode="json"), params=params
                )
                resp.raise_for_status()
                system_output = resp.json()

                # Score
                tps = evaluator.score_instance(
                    task.output, system_output, task.target_schema
                )
                g_count = len(flatten_json(task.output))
                s_count = len(flatten_json(system_output))
                status = "PASS"

            except Exception as e:
                # Penalize failures with 0 score
                status = "FAIL"
                tps = 0.0
                s_count = 0
                g_count = len(flatten_json(task.output)) if task else 0
                console.print(f"[bold red]Error processing {file_path.name}:[/bold red] {e}")

            tps_list.append(tps)
            gold_counts.append(g_count)
            system_counts.append(s_count)

            if task:
                results_table.add_row(
                    task.id, f"{tps:.2f}", str(g_count), str(s_count), 
                    f"[green]{status}[/green]" if status == "PASS" else f"[red]{status}[/red]"
                )
                individual_results.append({
                    "task_id": task.id,
                    "tps": tps,
                    "gold_keys": g_count,
                    "system_keys": s_count,
                    "status": status
                })

    # 3. Calculate final metrics
    metrics = evaluator.calculate_metrics(tps_list, gold_counts, system_counts)

    console.print(results_table)

    summary_table = Table(title="Aggregate Metrics", show_header=False)
    summary_table.add_row("Precision", f"{metrics['precision']:.4f}")
    summary_table.add_row("Recall", f"{metrics['recall']:.4f}")
    summary_table.add_row("Micro-F1", f"[bold green]{metrics['f1']:.4f}[/bold green]")
    console.print(summary_table)

    # 4. Export JSON Report
    if output:
        report = {
            "participant": participant_info,
            "config": {
                "model": model,
                "pipeline": pipeline,
                "data_source": str(data.absolute())
            },
            "metrics": metrics,
            "tasks": individual_results
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"\n[bold blue]Report saved to {output}[/bold blue]")


@app.command()
def leaderboard(
    results_dir: Path = typer.Argument(
        Path("results"), help="Directory containing evaluation JSON reports"
    ),
    plain: bool = typer.Option(False, "--plain", help="Output in plain Markdown format"),
):
    """
    Aggregates evaluation reports and displays a leaderboard.
    Groups results by (Model, Dataset) and ranks by Micro-F1.
    """
    if not results_dir.is_dir():
        console.print(f"[bold red]Error: {results_dir} is not a directory.[/bold red]")
        raise typer.Exit(1)

    reports = []
    for file_path in results_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Robustness: Validate required fields
            if not all(k in data for k in ["participant", "config", "metrics"]):
                continue

            reports.append(
                {
                    "team": data["participant"].get("team_name", "Unknown"),
                    "model": data["config"].get("model", "Unknown"),
                    "pipeline": data["config"].get("pipeline", "Unknown"),
                    "dataset": Path(data["config"].get("data_source", "Unknown")).name,
                    "f1": data["metrics"].get("f1", 0.0),
                }
            )
        except Exception:
            # Skip malformed or incompatible JSONs
            continue

    if not reports:
        console.print("[yellow]No valid evaluation reports found.[/yellow]")
        return

    # Grouping by (Model, Dataset)
    groups = defaultdict(list)
    for r in reports:
        groups[(r["model"], r["dataset"])].append(r)

    for (model, dataset), entries in groups.items():
        # Sort all entries by F1 descending
        entries.sort(key=lambda x: x["f1"], reverse=True)

        # Calculate Best-per-team
        best_per_team = {}
        for e in entries:
            if e["team"] not in best_per_team:
                best_per_team[e["team"]] = e
        best_entries = sorted(
            best_per_team.values(), key=lambda x: x["f1"], reverse=True
        )

        if plain:
            # Manual Markdown output
            print(f"## Leaderboard: {dataset} (Model: {model})")
            for title, data_list in [
                ("Best per Team", best_entries),
                ("All Pipelines", entries),
            ]:
                print(f"\n### {title}")
                print("| Rank | Team | Pipeline | Micro-F1 |")
                print("|---:|:---|:---|---:|")
                for i, e in enumerate(data_list, 1):
                    print(f"| {i} | {e['team']} | {e['pipeline']} | {e['f1']:.4f} |")
            print("\n")
        else:
            # Rich Terminal output
            console.rule(
                f"[bold green]Leaderboard: {dataset} (Model: {model})[/bold green]"
            )
            for title, data_list in [
                ("Best per Team", best_entries),
                ("All Pipelines", entries),
            ]:
                table = Table(title=title, box=None)
                table.add_column("Rank", justify="right", style="cyan")
                table.add_column("Team", style="magenta")
                table.add_column("Pipeline")
                table.add_column("Micro-F1", justify="right", style="bold green")
                for i, e in enumerate(data_list, 1):
                    table.add_row(str(i), e["team"], e["pipeline"], f"{e['f1']:.4f}")
                console.print(table)
            console.print("\n")


if __name__ == "__main__":
    app()
