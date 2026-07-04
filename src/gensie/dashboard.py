"""Rich live dashboard for `gensie eval-full`.

A single-window layout that updates in place while the sweep runs:

  ┌──────── eval-full · http://… · concurrency=16 · parallel=4 ────────┐
  │ Phase: salamandra7b (2/3)              Total ████░░ 8/30  ETA 1h23m │
  ├────────────────────────────────────────────────────────────────────┤
  │ driller     enriched-schema-rag      [████░░] 87/145 (60%) ETA 4m  │
  │             best F1: 0.7821                                        │
  │ sesml       extraction               [██░░░░] 58/145 (40%) ETA 6m  │
  │             best F1: 0.7820                                        │
  │ krishan     english_fewshot          [██████] DONE  F1=0.7805      │
  │ franrodrigo baseline                 [█████░] 142/145 (98%) ETA 5s │
  ├──────────────── Leaderboard (live, top 3) ─────────────────────────┤
  │ 1. krishan          english_fewshot       F1 0.7821 gap 32.3%     │
  │ 2. sesml            adaptive              F1 0.7815 gap 31.8%     │
  │ 3. driller          mixed-extractors-…    F1 0.7662 gap 28.9%     │
  └────────────────────────────────────────────────────────────────────┘

Per-team progress is sourced from each team's docker container logs
(`docker logs --since=<pipeline_start> <container> | grep -c 'POST /run'`).
Team runner threads update a shared `DashboardState` at boot / pipeline-start /
pipeline-end milestones; a background refresher polls container logs every
1.5 s for the in-pipeline instance count.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, ProgressColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


INSTANCES_PER_PIPELINE = 145


def _fmt_dur(s: float) -> str:
    s = int(s)
    if s >= 3600:
        return f"{s//3600}h{(s%3600)//60:02d}m"
    if s >= 60:
        return f"{s//60}m{s%60:02d}s"
    return f"{s}s"


# ───────────────────────── State ─────────────────────────


@dataclass
class TeamPhase:
    """One (team, model) cell — covers all that team's pipelines on that model."""
    slug: str
    model_id: str
    container_name: Optional[str] = None
    status: str = "pending"  # pending | booting | running | done | failed
    current_pipeline: Optional[str] = None
    pipeline_start_ts: Optional[float] = None
    processed: int = 0
    pipelines_done: List[Tuple[str, str, Optional[float]]] = field(default_factory=list)
    # ^ (pipeline, status, f1_or_None)
    pipelines_total: int = 0
    err: Optional[str] = None


@dataclass
class DashboardState:
    server_url: str
    concurrency: int
    parallel_teams: int
    output_dir: Path
    started_ts: float = field(default_factory=time.time)
    current_phase: Optional[str] = None  # model_id
    total_target: int = 0  # total (team, pipeline, model) expected
    total_done: int = 0    # PASS + SKIPPED + EVAL_FAIL count

    # team_state[(slug, model_id)] = TeamPhase
    team_state: Dict[Tuple[str, str], TeamPhase] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_total(self, target: int) -> None:
        with self.lock:
            self.total_target = target

    def set_phase(self, model_id: str) -> None:
        with self.lock:
            self.current_phase = model_id

    def get_or_create(self, slug: str, model_id: str) -> TeamPhase:
        with self.lock:
            key = (slug, model_id)
            if key not in self.team_state:
                self.team_state[key] = TeamPhase(slug=slug, model_id=model_id)
            return self.team_state[key]

    def increment_done(self) -> None:
        with self.lock:
            self.total_done += 1


# ───────────────────────── Polling thread ─────────────────────────


def _docker_post_count(container: str, since_ts: float) -> int:
    """Count `POST /run` lines emitted since `since_ts` in the container's docker logs."""
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since_ts))
    try:
        proc = subprocess.run(
            ["docker", "logs", "--since", iso, container],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout.count("POST /run") + proc.stderr.count("POST /run")
    except Exception:
        return 0


def _poller(state: DashboardState, stop_event: threading.Event, interval: float = 1.5) -> None:
    """Refresh `processed` counts on each running team from docker logs."""
    while not stop_event.is_set():
        with state.lock:
            running = [
                tp for tp in state.team_state.values()
                if tp.status == "running" and tp.container_name and tp.pipeline_start_ts
            ]
        for tp in running:
            n = _docker_post_count(tp.container_name, tp.pipeline_start_ts)
            with state.lock:
                tp.processed = n
        stop_event.wait(interval)


# ───────────────────────── Renderer ─────────────────────────


def _bar(processed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "░" * width + "]"
    f = min(width, processed * width // total)
    return "[" + "█" * f + "░" * (width - f) + "]"


def _render(state: DashboardState) -> Panel:
    with state.lock:
        # Header line
        elapsed = time.time() - state.started_ts
        total_pct = (state.total_done * 100 // state.total_target) if state.total_target else 0
        # Total ETA: completed / elapsed × remaining
        if state.total_done > 0 and state.total_target > state.total_done:
            rate = state.total_done / elapsed
            remaining = (state.total_target - state.total_done) / rate
            total_eta = _fmt_dur(remaining)
        else:
            total_eta = "—"

        header_t = Table.grid(expand=True)
        header_t.add_column(ratio=1)
        header_t.add_column(justify="right")
        phase_str = state.current_phase or "—"
        header_t.add_row(
            f"[dim]server:[/dim] {state.server_url}  [dim]concurrency:[/dim] {state.concurrency}  "
            f"[dim]parallel_teams:[/dim] {state.parallel_teams}",
            f"[dim]elapsed:[/dim] {_fmt_dur(elapsed)}",
        )
        header_t.add_row(
            f"[bold]Phase:[/bold] [magenta]{phase_str}[/magenta]",
            f"[bold]Total[/bold] {_bar(state.total_done, state.total_target, 24)} "
            f"{state.total_done}/{state.total_target} ({total_pct}%) "
            f"ETA [yellow]{total_eta}[/yellow]",
        )

        # Per-team table
        team_t = Table.grid(expand=True, padding=(0, 1))
        team_t.add_column(width=14)         # team slug
        team_t.add_column(width=36)         # pipeline name
        team_t.add_column(width=32)         # progress bar + count
        team_t.add_column(justify="right")  # F1 / status

        for (slug, model_id), tp in sorted(state.team_state.items()):
            # F1 best across this team's done pipelines
            done_f1s = [f1 for _, st, f1 in tp.pipelines_done if st == "PASS" and f1 is not None]
            best_f1 = max(done_f1s) if done_f1s else None
            last_f1 = done_f1s[-1] if done_f1s else None

            if tp.status == "pending":
                team_t.add_row(f"[dim]{slug}[/dim]", "[dim]pending[/dim]", "", "")
                continue
            if tp.status == "booting":
                team_t.add_row(f"[yellow]{slug}[/yellow]", "[yellow]booting…[/yellow]", "", "")
                continue
            if tp.status == "failed":
                team_t.add_row(
                    f"[red]{slug}[/red]",
                    f"[red]failed:[/red] {(tp.err or '')[:30]}",
                    "", "",
                )
                continue
            if tp.status == "done":
                f1_s = f"[green]done[/green]  best F1 [bold green]{best_f1:.4f}[/bold green]" if best_f1 is not None else "[green]done[/green]"
                # Summary line: pipelines list
                pl_done = ",".join(
                    (f"[green]{p}[/green]" if st == "PASS" else f"[red]{p}[/red]")
                    for p, st, _ in tp.pipelines_done
                )
                team_t.add_row(f"[green]{slug}[/green]", pl_done[:34], "", f1_s)
                continue

            # running
            pipe = tp.current_pipeline or "—"
            processed = tp.processed
            bar = _bar(processed, INSTANCES_PER_PIPELINE, 22)
            pct = processed * 100 // INSTANCES_PER_PIPELINE if INSTANCES_PER_PIPELINE else 0
            pelapsed = time.time() - (tp.pipeline_start_ts or time.time())
            if processed > 0 and processed < INSTANCES_PER_PIPELINE:
                eta = _fmt_dur(pelapsed * (INSTANCES_PER_PIPELINE - processed) / processed)
            else:
                eta = "—"
            f1_s = ""
            if best_f1 is not None:
                f1_s = f"best [bold]{best_f1:.4f}[/bold]"
            team_t.add_row(
                f"[cyan]{slug}[/cyan]",
                pipe[:34],
                f"{bar} {processed:>3}/{INSTANCES_PER_PIPELINE} ETA {eta}",
                f1_s,
            )

        # Live leaderboard: top 3 (team, pipeline, F1) by F1
        rows = []
        for (slug, _mid), tp in state.team_state.items():
            for p, st, f1 in tp.pipelines_done:
                if st == "PASS" and f1 is not None:
                    rows.append((slug, p, f1))
        rows.sort(key=lambda r: r[2], reverse=True)
        top3 = rows[:3]
        lb = Table.grid(padding=(0, 2))
        lb.add_column(width=4); lb.add_column(width=14)
        lb.add_column(width=32); lb.add_column(justify="right")
        if not top3:
            lb.add_row("", "[dim]no PASS rows yet[/dim]", "", "")
        else:
            for i, (slug, pipe, f1) in enumerate(top3, 1):
                lb.add_row(f"{i}.", slug, pipe[:30], f"F1 [bold green]{f1:.4f}[/bold green]")

        body = Group(
            header_t,
            Text(""),
            team_t,
            Text(""),
            Panel(lb, title="Leaderboard (live)", title_align="left", expand=True),
        )
        return Panel(body, title="GenSIE eval-full", title_align="left")
