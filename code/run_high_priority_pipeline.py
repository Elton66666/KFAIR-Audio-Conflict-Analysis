import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/root/autodl-tmp/kfair")
CODE = ROOT / "code"
OUT = ROOT / "outputs"
STATE = OUT / "high_priority_pipeline_state.json"


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def write_state(stage, status, **extra):
    STATE.write_text(
        json.dumps(
            {"updated_at": datetime.now().isoformat(), "stage": stage, "status": status, **extra},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(stage, *arguments):
    write_state(stage, "running", command=list(arguments))
    log(f"START {stage}: {' '.join(map(str, arguments))}")
    subprocess.run(arguments, cwd=CODE, check=True, env=os.environ.copy())
    write_state(stage, "completed")
    log(f"DONE {stage}")


def main():
    os.environ.update(
        {
            "PYTHONPATH": str(CODE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(ROOT / "cache/huggingface"),
            "TMPDIR": "/root/autodl-tmp/tmp",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
        }
    )
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    pid = int((OUT / "actor_cv6.pid").read_text().strip())
    write_state("actor_cv6_wait", "running", actor_cv6_pid=pid)
    while process_alive(pid):
        completed = 0
        summary = OUT / "actor_cv6_run_summary.jsonl"
        if summary.exists():
            completed = sum(1 for line in summary.read_text(encoding="utf-8").splitlines() if line.strip())
        log(f"WAIT actor_cv6 pid={pid} completed={completed}/54")
        time.sleep(30)
    summary = OUT / "actor_cv6_run_summary.jsonl"
    completed = sum(1 for line in summary.read_text(encoding="utf-8").splitlines() if line.strip())
    if completed != 54:
        write_state("actor_cv6", "failed", completed=completed, expected=54)
        raise RuntimeError(f"actor CV exited with only {completed}/54 completed runs")
    write_state("actor_cv6", "completed", completed=completed)
    run("actor_cv6_analysis", sys.executable, "analyze_actor_cv6.py")
    run("qwen_behavior", sys.executable, "run_qwen_behavioral_replication.py", "--overwrite")
    run("qwen_adapter", sys.executable, "run_qwen_adapter_replication.py")
    run("qwen_analysis", sys.executable, "analyze_qwen_replication.py")
    write_state("all_high_priority_experiments", "completed")
    log("ALL HIGH-PRIORITY EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_state("pipeline", "failed", error=repr(error))
        log(f"FAILED {error!r}")
        raise
