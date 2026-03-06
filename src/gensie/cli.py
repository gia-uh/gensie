import typer
import uvicorn
import httpx
from pathlib import Path
from typing import Optional
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
):
    """Evaluates the agent against a local dataset."""
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

    results_table = Table(title=f"Evaluation Results (Pipeline: {pipeline})")
    results_table.add_column("Task ID", style="cyan")
    results_table.add_column("TPS", justify="right")
    results_table.add_column("Gold Keys", justify="right")
    results_table.add_column("System Keys", justify="right")

    params = {"pipeline": pipeline}
    if model:
        params["model"] = model

    with httpx.Client(timeout=60.0) as client:
        # Check /info first
        try:
            info_resp = client.get(f"{url}/info")
            info_resp.raise_for_status()
            console.print(f"Auditing Team: [bold magenta]{info_resp.json()['team_name']}[/bold magenta]")
        except Exception:
            console.print("[yellow]Could not retrieve participant info.[/yellow]")

        for file_path in track(json_files, description="Processing tasks..."):
            try:
                task = Task.load(file_path)
                # Call agent
                resp = client.post(
                    f"{url}/run", 
                    json=task.model_dump(mode="json"),
                    params=params
                )
                resp.raise_for_status()
                system_output = resp.json()

                # Score
                tps = evaluator.score_instance(
                    task.output, system_output, task.target_schema
                )
                g_count = len(flatten_json(task.output))
                s_count = len(flatten_json(system_output))

                tps_list.append(tps)
                gold_counts.append(g_count)
                system_counts.append(s_count)

                results_table.add_row(task.id, f"{tps:.2f}", str(g_count), str(s_count))
            except Exception as e:
                console.print(f"[yellow]Skipping {file_path.name}: {e}[/yellow]")

    # Calculate final metrics
    metrics = evaluator.calculate_metrics(tps_list, gold_counts, system_counts)

    console.print(results_table)

    summary_table = Table(title="Aggregate Metrics", show_header=False)
    summary_table.add_row("Precision", f"{metrics['precision']:.4f}")
    summary_table.add_row("Recall", f"{metrics['recall']:.4f}")
    summary_table.add_row("Micro-F1", f"[bold green]{metrics['f1']:.4f}[/bold green]")
    console.print(summary_table)


if __name__ == "__main__":
    app()
