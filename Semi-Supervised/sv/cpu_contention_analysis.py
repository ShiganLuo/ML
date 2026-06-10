"""CPU contention analysis for running Python/joblib processes.

Scans for Python processes with loky workers, measures per-process
CPU utilization, and estimates the runtime impact of contention.

Usage:
    python cpu_contention_analysis.py
    python cpu_contention_analysis.py --interval 10 --samples 6
    python cpu_contention_analysis.py --pid 110667
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    cpu_pct: float
    cpu_time_sec: float
    elapsed_sec: float
    cmd: str
    is_worker: bool = False
    parent_pid: Optional[int] = None


@dataclass
class TaskGroup:
    parent_pid: int
    cmd: str
    workers: List[ProcessInfo] = field(default_factory=list)

    @property
    def n_workers(self) -> int:
        return len(self.workers)

    @property
    def total_cpu_pct(self) -> float:
        return sum(w.cpu_pct for w in self.workers)

    @property
    def total_cpu_time_sec(self) -> float:
        return sum(w.cpu_time_sec for w in self.workers)

    @property
    def avg_cpu_pct_per_worker(self) -> float:
        return self.total_cpu_pct / max(self.n_workers, 1)

    @property
    def wall_time_sec(self) -> float:
        if self.workers:
            return max(w.elapsed_sec for w in self.workers)
        return 0.0

    @property
    def effective_cores(self) -> float:
        return self.total_cpu_pct / 100.0


def parse_time_str(time_str: str) -> float:
    time_str = time_str.strip()
    parts = time_str.split("-")
    days = 0
    if len(parts) == 2:
        days = int(parts[0])
        time_str = parts[1]
    segments = time_str.split(":")
    if len(segments) == 3:
        h, m, s = int(segments[0]), int(segments[1]), int(segments[2])
    elif len(segments) == 2:
        h, m, s = 0, int(segments[0]), int(segments[1])
    else:
        return 0.0
    return days * 86400 + h * 3600 + m * 60 + s


def collect_processes(target_pids: Optional[List[int]] = None) -> List[ProcessInfo]:
    cmd = ["ps", "-eo", "pid,ppid,pcpu,cputime,etime,args", "--no-headers"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    processes = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        cpu_pct = float(parts[2])
        cpu_time = parse_time_str(parts[3])
        elapsed = parse_time_str(parts[4])
        cmd_str = parts[5]
        if "python" not in cmd_str.lower():
            continue
        if target_pids and pid not in target_pids and ppid not in target_pids:
            continue
        is_worker = "loky" in cmd_str or "popen_loky" in cmd_str
        processes.append(ProcessInfo(
            pid=pid, ppid=ppid, cpu_pct=cpu_pct,
            cpu_time_sec=cpu_time, elapsed_sec=elapsed,
            cmd=cmd_str, is_worker=is_worker,
            parent_pid=ppid if is_worker else None,
        ))
    return processes


def group_by_task(processes: List[ProcessInfo]) -> List[TaskGroup]:
    parents = {p.pid: p for p in processes if not p.is_worker}
    children = [p for p in processes if p.is_worker]
    groups: Dict[int, TaskGroup] = {}
    for pid in parents:
        groups[pid] = TaskGroup(parent_pid=pid, cmd=parents[pid].cmd, workers=[])
    for child in children:
        parent_pid = child.parent_pid
        if parent_pid in groups:
            groups[parent_pid].workers.append(child)
    return list(groups.values())


def measure_cpu_samples(target_pids=None, interval=5.0, samples=3):
    snapshots = []
    for i in range(samples):
        snapshots.append(collect_processes(target_pids))
        if i < samples - 1:
            time.sleep(interval)
    return snapshots


def estimate_contention_impact(groups: List[TaskGroup], n_cores: int):
    total_effective = sum(g.effective_cores for g in groups)
    contention_ratio = total_effective / max(n_cores, 1)
    results = {"n_cores": n_cores, "total_effective_cores": round(total_effective, 1),
               "contention_ratio": round(contention_ratio, 2), "tasks": []}
    for group in groups:
        if not group.workers:
            continue
        n_workers = group.n_workers
        eff_per_worker = group.avg_cpu_pct_per_worker / 100.0
        slowdown = 1.0 / max(eff_per_worker, 0.01)
        total_cpu = group.total_cpu_time_sec
        wall = group.wall_time_sec
        cpu_eff = total_cpu / max(wall * n_workers, 1) if wall > 0 else 0
        results["tasks"].append({
            "parent_pid": group.parent_pid,
            "cmd_short": group.cmd[:120],
            "n_workers": n_workers,
            "cpu_pct_per_worker": round(group.avg_cpu_pct_per_worker, 1),
            "total_cpu_pct": round(group.total_cpu_pct, 1),
            "effective_cores": round(group.effective_cores, 1),
            "wall_time_min": round(wall / 60, 1),
            "cpu_time_total_min": round(total_cpu / 60, 1),
            "slowdown_factor": round(slowdown, 1),
            "cpu_efficiency": round(cpu_eff * 100, 1),
        })
    return results


def generate_report(results: Dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CPU Contention Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"CPU cores available:    {results['n_cores']}")
    lines.append(f"Total effective usage:  {results['total_effective_cores']} cores")
    lines.append(f"Contention ratio:      {results['contention_ratio']}x")
    lines.append("")

    n_tasks = len(results["tasks"])
    if results["contention_ratio"] > 1.0:
        lines.append(f"  !! {results['total_effective_cores']} cores requested on {results['n_cores']} available")
        lines.append(f"     Each task gets ~{results['n_cores'] / max(n_tasks, 1):.1f} cores instead of full capacity")
        lines.append("")

    lines.append("-" * 70)
    lines.append("Per-Task Breakdown")
    lines.append("-" * 70)

    for t in results["tasks"]:
        lines.append("")
        lines.append(f"  Task PID {t['parent_pid']}:")
        lines.append(f"    Command:          {t['cmd_short']}")
        lines.append(f"    Workers:          {t['n_workers']}")
        lines.append(f"    CPU/worker:       {t['cpu_pct_per_worker']}%")
        lines.append(f"    Total CPU:        {t['total_cpu_pct']}%  ({t['effective_cores']} cores)")
        lines.append(f"    Wall time:        {t['wall_time_min']} min")
        lines.append(f"    CPU time (total): {t['cpu_time_total_min']} min")
        lines.append(f"    Slowdown factor:  {t['slowdown_factor']}x")
        lines.append(f"    CPU efficiency:   {t['cpu_efficiency']}%")

    lines.append("")
    lines.append("-" * 70)
    lines.append("Impact Summary")
    lines.append("-" * 70)

    if n_tasks == 0:
        lines.append("  No Python tasks detected.")
    elif n_tasks == 1:
        t = results["tasks"][0]
        lines.append(f"  Single task with {t['n_workers']} workers")
        lines.append(f"  Effective: {t['effective_cores']} cores / {results['n_cores']} available")
        if t["slowdown_factor"] > 1.5:
            lines.append(f"  !! Workers competing: ~{t['slowdown_factor']}x slower than ideal")
    else:
        lines.append(f"  {n_tasks} tasks running concurrently, {sum(t['n_workers'] for t in results['tasks'])} total workers")
        total_eff = sum(t["effective_cores"] for t in results["tasks"])
        lines.append(f"  Combined usage: {total_eff:.1f} / {results['n_cores']} cores")
        avg_slowdown = sum(t["slowdown_factor"] for t in results["tasks"]) / n_tasks
        lines.append(f"  Average slowdown: {avg_slowdown:.1f}x")
        lines.append("")
        if n_tasks > 1:
            seq_time = sum(t["cpu_time_total_min"] / t["n_workers"] for t in results["tasks"])
            par_time = max(t["wall_time_min"] for t in results["tasks"])
            lines.append(f"  Sequential estimate: {seq_time:.0f} min total ({seq_time / n_tasks:.0f} min/task)")
            lines.append(f"  Parallel actual:     {par_time:.0f} min (and counting)")

    lines.append("")
    lines.append("-" * 70)
    lines.append("Recommendations")
    lines.append("-" * 70)

    if n_tasks > 1:
        lines.append("  1. Run tasks sequentially instead of concurrently")
        lines.append(f"  2. Each task will use all {results['n_cores']} cores -> ~{n_tasks}x faster per task")
        for t in results["tasks"]:
            est = t["cpu_time_total_min"] / t["n_workers"]
            lines.append(f"     PID {t['parent_pid']}: ~{est:.0f} min (currently {t['wall_time_min']}+ min)")

    for t in results["tasks"]:
        if t["n_workers"] > results["n_cores"]:
            lines.append(f"  3. Task PID {t['parent_pid']}: reduce workers from {t['n_workers']} to {results['n_cores']}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CPU contention analysis")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--pid", type=int, action="append", default=None)
    args = parser.parse_args()

    n_cores = os.cpu_count() or 1
    print(f"Detected {n_cores} CPU cores")
    print(f"Taking {args.samples} samples (interval={args.interval}s)...")
    print()

    snapshots = measure_cpu_samples(args.pid, args.interval, args.samples)
    processes = snapshots[-1]
    groups = group_by_task(processes)

    if not groups:
        print("No Python tasks with workers detected.")
        return

    results = estimate_contention_impact(groups, n_cores)
    print(generate_report(results))

    if len(snapshots) > 1:
        print()
        print("-" * 70)
        print("CPU Trend (per sample)")
        print("-" * 70)
        for i, snap in enumerate(snapshots):
            total = sum(p.cpu_pct for p in snap)
            n_w = sum(1 for p in snap if p.is_worker)
            print(f"  Sample {i+1}: {n_w} workers, total CPU {total:.1f}%")


if __name__ == "__main__":
    main()
