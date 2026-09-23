from __future__ import annotations

import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

from autods_core import json_default, load_demo, replay_states, slim_state

REPLAY_STEP_SEC = 0.9


def emit(out, kind: str, started: float, **payload) -> None:
    line = {"type": kind, "t": round(time.time() - started, 3), "ts": time.time(), **payload}
    out.write(json.dumps(line, default=json_default) + "\n")
    out.flush()


def run_replay(out, started: float) -> None:
    demo = load_demo()
    for state in replay_states(demo):
        time.sleep(REPLAY_STEP_SEC)
        emit(out, "state", started, state=state)
    emit(out, "done", started, result=demo["result"])


def run_live(run_dir: Path, out, started: float) -> None:
    import pandas as pd

    from experiment_common import make_split
    from pipeline import build_result_from_state, run_agentic_streaming

    cfg = json.loads((run_dir / "config.json").read_text())
    df = pd.read_csv(run_dir / "data.csv")
    (run_dir / "data.csv").unlink(missing_ok=True)
    target = cfg["target_column"]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    train_df, test_df = make_split(df, target, cfg["seed"])

    last_state = None
    for state in run_agentic_streaming(
        train_df, test_df, target, cfg["positive_label"], cfg["file_name"], cfg["seed"],
        max_iterations=cfg["max_iterations"], performance_target=cfg["performance_target"],
    ):
        last_state = state
        emit(out, "state", started, state=slim_state(state))

    result = build_result_from_state(
        last_state, cfg["file_name"], cfg["seed"], cfg["max_iterations"], None, cfg["performance_target"],
    )
    result.runtime_wallclock_sec = round(time.time() - started, 3)
    emit(out, "done", started, result=dataclasses.asdict(result))


def main() -> None:
    mode, run_dir = sys.argv[1], Path(sys.argv[2])
    started = time.time()
    with open(run_dir / "events.jsonl", "a", buffering=1) as out:
        try:
            if mode == "replay":
                run_replay(out, started)
            else:
                run_live(run_dir, out, started)
        except Exception as exc:
            emit(out, "error", started, message=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])


if __name__ == "__main__":
    main()
