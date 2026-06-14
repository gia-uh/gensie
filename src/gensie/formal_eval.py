"""Full-sweep orchestration: every team × every pipeline × every model.

The driver assumes the model server (vLLM, SGLang, LM Studio, …) is already
serving each configured model at `config.server_url`. Per phase it composes
up each team's agent container (in parallel up to `parallel_teams`), runs
`gensie eval --concurrency N` against it, writes a per-(team, pipeline,
model) report, and appends a row to `status.jsonl`.

Driven by a YAML config. See `formal-eval-config.yaml` for the schema.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from rich.console import Console
from rich.progress import Progress

console = Console()


# ───────────────────────── Config dataclasses ─────────────────────────


@dataclass
class TeamSpec:
    slug: str
    pipelines: List[str] = field(default_factory=list)  # empty → auto-discover via /info
    overrides: List[str] = field(default_factory=list)  # paths relative to runtime_dir


@dataclass
class ModelSpec:
    id: str
    served_name: str  # what the OpenAI-compatible server returns as `id`
    only_teams: List[str] = field(default_factory=list)  # if non-empty, restrict


@dataclass
class EvalConfig:
    server_url: str
    data: Path
    participants_dir: Path
    runtime_dir: Path
    output_dir: Path
    teams: List[TeamSpec]
    models: List[ModelSpec]
    base_port: int = 9100
    request_timeout_s: int = 300
    pipeline_budget_s: int = 9000  # hard cap per (team, pipeline)
    gensie_bin: str = "gensie"  # path to the binary (uv run gensie / venv-installed)

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalConfig":
        raw = yaml.safe_load(path.read_text())
        base = path.parent.resolve()
        # Resolve relative paths against the config file's directory.
        return cls(
            server_url=raw["server_url"],
            data=(base / raw["data"]).resolve(),
            participants_dir=Path(os.path.expanduser(raw["participants_dir"])).resolve(),
            runtime_dir=(base / raw["runtime_dir"]).resolve(),
            output_dir=(base / raw["output_dir"]).resolve(),
            teams=[TeamSpec(**t) for t in raw["teams"]],
            models=[ModelSpec(**m) for m in raw["models"]],
            base_port=raw.get("base_port", 9100),
            request_timeout_s=raw.get("request_timeout_s", 300),
            pipeline_budget_s=raw.get("pipeline_budget_s", 9000),
            gensie_bin=raw.get("gensie_bin", "gensie"),
        )


# ───────────────────────── Helpers ─────────────────────────


def _verify_model(server_url: str, served_name: str, timeout_s: int = 10) -> bool:
    """GET /v1/models and check that `served_name` is among the listed ids."""
    try:
        r = httpx.get(f"{server_url.rstrip('/v1')}/v1/models", timeout=timeout_s)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        return served_name in ids
    except Exception as e:
        console.print(f"[red]model verify failed:[/red] {e}")
        return False


def _compose_files(cfg: EvalConfig, team: TeamSpec) -> List[str]:
    """Return the -f flag list for `docker compose` for this team."""
    files = ["-f", "docker-compose.yml"]
    files.extend(["-f", str(cfg.runtime_dir / "extra-hosts.yml")])
    files.extend(["-f", str(cfg.runtime_dir / "extra-port.yml")])
    for ovr in team.overrides:
        files.extend(["-f", str(cfg.runtime_dir / ovr)])
    return files


def _compose_up(cfg: EvalConfig, team: TeamSpec, host_port: int, log_path: Path) -> bool:
    proj = f"ef-{team.slug}-{host_port}"
    clone_dir = cfg.participants_dir / team.slug
    env = {
        **os.environ,
        "HOST_PORT": str(host_port),
        "OPENAI_BASE_URL": cfg.server_url,
        "OPENAI_API_KEY": "lm-studio",
        "PARTICIPANT_PATH": "gensie.baseline.OfficialParticipant",
    }
    files = _compose_files(cfg, team)
    # Tear down any prior project with the same name.
    subprocess.run(
        ["docker", "compose", *files, "--project-name", proj, "down"],
        cwd=clone_dir, env=env, capture_output=True,
    )
    with open(log_path, "w") as f:
        rc = subprocess.run(
            ["docker", "compose", *files, "--project-name", proj, "up", "-d"],
            cwd=clone_dir, env=env, stdout=f, stderr=subprocess.STDOUT,
        ).returncode
    return rc == 0


def _compose_down(cfg: EvalConfig, team: TeamSpec, host_port: int) -> None:
    proj = f"ef-{team.slug}-{host_port}"
    clone_dir = cfg.participants_dir / team.slug
    subprocess.run(
        ["docker", "compose", "--project-name", proj, "down"],
        cwd=clone_dir, capture_output=True,
    )


def _wait_for_info(host_port: int, timeout_s: int = 120) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{host_port}/info", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _fetch_pipelines(host_port: int) -> List[str]:
    try:
        r = httpx.get(f"http://127.0.0.1:{host_port}/info", timeout=5)
        return [p["name"] for p in r.json().get("pipelines", [])]
    except Exception:
        return []


def _emit_status(status_path: Path, row: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with status_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_one_pipeline(
    cfg: EvalConfig,
    team: TeamSpec,
    model: ModelSpec,
    pipeline: str,
    host_port: int,
    concurrency: int,
    limit: Optional[int],
    reports_dir: Path,
    logs_dir: Path,
) -> Dict[str, Any]:
    """Invoke gensie eval as a subprocess; return a status row dict."""
    report = reports_dir / f"{team.slug}--{pipeline}--{model.id}.json"
    log = logs_dir / f"{team.slug}--{pipeline}--{model.id}.eval.log"
    cmd = [
        cfg.gensie_bin, "eval",
        "--data", str(cfg.data),
        "--url", f"http://127.0.0.1:{host_port}",
        "--pipeline", pipeline,
        "--model", model.served_name,
        "--output", str(report),
        "--request-timeout-s", str(cfg.request_timeout_s),
        "--concurrency", str(concurrency),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            timeout=cfg.pipeline_budget_s + 60,  # +60s grace
        ).returncode
    elapsed = int(time.time() - t0)

    if rc != 0:
        return {
            "slug": team.slug, "pipeline": pipeline,
            "model": model.served_name, "thinking": "na",
            "report_tag": model.id, "status": "EVAL_FAIL",
            "err": f"exit {rc}", "wall_s": elapsed,
        }
    try:
        data = json.loads(report.read_text())
        m = data["metrics"]
        return {
            "slug": team.slug, "pipeline": pipeline,
            "model": model.served_name, "thinking": "na",
            "report_tag": model.id, "status": "PASS",
            "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "avg_time_s": data["timing"]["avg_elapsed_s"],
            "max_time_s": data["timing"]["max_elapsed_s"],
            "n": data["timing"]["n"],
            "avg_tokens": data["token_usage"]["avg_total_per_instance"],
            "wall_s": elapsed,
        }
    except Exception as e:
        return {
            "slug": team.slug, "pipeline": pipeline,
            "model": model.served_name, "thinking": "na",
            "report_tag": model.id, "status": "EVAL_FAIL",
            "err": f"report-parse: {e}", "wall_s": elapsed,
        }


def _run_one_team(
    cfg: EvalConfig,
    team: TeamSpec,
    model: ModelSpec,
    host_port: int,
    concurrency: int,
    limit: Optional[int],
    status_path: Path,
    status_lock: threading.Lock,
    reports_dir: Path,
    logs_dir: Path,
) -> None:
    """End-to-end for one (team, model): compose up, run all pipelines, compose down."""
    boot_log = logs_dir / f"{team.slug}--{model.id}.boot.log"
    if not _compose_up(cfg, team, host_port, boot_log):
        _emit_status(status_path, {
            "slug": team.slug, "pipeline": "-", "model": model.served_name,
            "thinking": "na", "report_tag": model.id, "status": "BOOT_FAIL",
            "err": "compose up failed",
        }, status_lock)
        return

    if not _wait_for_info(host_port):
        _emit_status(status_path, {
            "slug": team.slug, "pipeline": "-", "model": model.served_name,
            "thinking": "na", "report_tag": model.id, "status": "BOOT_FAIL",
            "err": "no /info in 120s",
        }, status_lock)
        _compose_down(cfg, team, host_port)
        return

    pipelines = team.pipelines or _fetch_pipelines(host_port)
    if not pipelines:
        _emit_status(status_path, {
            "slug": team.slug, "pipeline": "-", "model": model.served_name,
            "thinking": "na", "report_tag": model.id, "status": "BOOT_FAIL",
            "err": "no pipelines reported",
        }, status_lock)
        _compose_down(cfg, team, host_port)
        return

    for p in pipelines:
        console.print(f"  [{team.slug}/{model.id}] {p}: starting")
        row = _run_one_pipeline(
            cfg, team, model, p, host_port, concurrency, limit, reports_dir, logs_dir,
        )
        _emit_status(status_path, row, status_lock)
        if row["status"] == "PASS":
            console.print(
                f"  [{team.slug}/{model.id}] {p}: [green]PASS[/green] "
                f"F1={row['f1']:.4f} wall={row['wall_s']}s"
            )
        else:
            console.print(
                f"  [{team.slug}/{model.id}] {p}: [red]{row['status']}[/red] "
                f"{row.get('err', '')[:60]}"
            )

    _compose_down(cfg, team, host_port)


# ───────────────────────── Public entry point ─────────────────────────


def run_eval_full(
    config: EvalConfig,
    concurrency: int = 16,
    parallel_teams: int = 1,
    limit: Optional[int] = None,
    only_models: Optional[List[str]] = None,
    only_teams: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Run the full evaluation: every (configured) model × every team × every pipeline."""
    cfg = config
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = cfg.output_dir / "reports"
    logs_dir = cfg.output_dir / "logs"
    reports_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    status_path = cfg.output_dir / "status.jsonl"
    status_lock = threading.Lock()

    models = cfg.models
    if only_models:
        models = [m for m in models if m.id in only_models]
    teams = cfg.teams
    if only_teams:
        teams = [t for t in teams if t.slug in only_teams]

    console.print(
        f"[bold]eval-full[/bold] · {len(models)} model(s) × {len(teams)} team(s) · "
        f"parallel_teams={parallel_teams} · concurrency={concurrency}"
    )
    console.print(f"  server: {cfg.server_url}")
    console.print(f"  output: {cfg.output_dir}")

    if dry_run:
        for m in models:
            console.print(f"\n[blue]Phase {m.id}[/blue] (served as {m.served_name})")
            for t in teams:
                if m.only_teams and t.slug not in m.only_teams:
                    continue
                console.print(f"  - {t.slug} (pipelines: {t.pipelines or 'auto'})")
        return

    for m in models:
        console.print(f"\n[bold blue]── Phase: {m.id} (served as {m.served_name}) ──[/bold blue]")
        if not _verify_model(cfg.server_url, m.served_name):
            console.print(f"[red]Skipping phase {m.id}: model not ready at {cfg.server_url}[/red]")
            continue

        phase_teams = [t for t in teams if not m.only_teams or t.slug in m.only_teams]

        # Run up to parallel_teams threads concurrently.
        sem = threading.Semaphore(parallel_teams)
        threads = []

        def _runner(team: TeamSpec, host_port: int) -> None:
            with sem:
                _run_one_team(
                    cfg, team, m, host_port, concurrency, limit,
                    status_path, status_lock, reports_dir, logs_dir,
                )

        for i, team in enumerate(phase_teams):
            host_port = cfg.base_port + (i % parallel_teams)
            t = threading.Thread(target=_runner, args=(team, host_port), daemon=False)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        console.print(f"[bold blue]── Phase {m.id} complete ──[/bold blue]")

    console.print("\n[bold green]== eval-full complete ==[/bold green]")
    console.print(f"  status: {status_path}")
    console.print(f"  reports: {reports_dir}")
