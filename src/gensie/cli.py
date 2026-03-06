import typer
import uvicorn
import httpx
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.progress import track
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


if __name__ == "__main__":
    app()
