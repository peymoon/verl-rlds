#!/usr/bin/env python3
"""Re-score saved validation JSONL dumps and build accuracy curves.

Old VLAA/ViRL runs sometimes logged shaped reward under the ``acc`` key.
This script recomputes true task accuracy from saved validation generations:

    output + gts (+ verifier_type for VLAA) -> fixed reward scorer -> acc

It writes one CSV row per validation step.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
VERL_SRC = REPO_ROOT / "verl"
if str(VERL_SRC) not in sys.path:
    sys.path.insert(0, str(VERL_SRC))

from verl.utils.reward_score import virl39k, vlaa  # noqa: E402


DEFAULT_ROOTS = [
    "/workspace/rl_data_selection/peyman/outputs/rollouts",
    "/workspace/rl_data_selection/peyman/outputs/virl39k/rollouts",
]

FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _step_from_file(path: Path, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def _infer_dataset(run_dir: Path, rows: list[dict[str, Any]], forced: str) -> str:
    if forced != "auto":
        return forced

    lower = str(run_dir).lower()
    if "virl39k" in lower:
        return "virl39k"

    verifier_types = {
        str(row.get("verifier_type", "")).lower()
        for row in rows[:50]
        if row.get("verifier_type") is not None
    }
    if verifier_types & {"open_ended", "iou", "mcq", "digit"}:
        return "vlaa"

    # ViRL is math-only; VLAA math-only subsets also work with the VLAA scorer.
    # Defaulting to VLAA outside virl39k paths is the least surprising behavior
    # for the existing outputs/rollouts directories.
    return "vlaa"


def _format_reward_text(output: Any) -> float:
    return 1.0 if FORMAT_RE.fullmatch(str(output)) else 0.0


def _score_from_logged(row: dict[str, Any], format_score: float) -> dict[str, float] | None:
    """Recover true acc from a saved shaped score when available.

    The old and new scorers both use:

        score = (1 - format_score) * acc + format_score * format

    So saved validation dumps that include ``score``/``reward`` can be converted
    exactly without rerunning math/open-ended/IoU verifiers.
    """
    score = _as_float(row.get("score"))
    if score is None:
        score = _as_float(row.get("reward"))
    if score is None:
        score = _as_float(row.get("acc"))
    if score is None:
        return None

    output = row.get("output", row.get("response", row.get("solution_str", "")))
    fmt = _format_reward_text(output)
    denom = max(1.0 - format_score, 1e-12)
    acc = (score - format_score * fmt) / denom
    acc = min(max(acc, 0.0), 1.0)
    if abs(acc - 0.0) < 1e-6:
        acc = 0.0
    elif abs(acc - 1.0) < 1e-6:
        acc = 1.0

    return {
        "acc": float(acc),
        "score": float(score),
        "format": float(fmt),
        "source": "logged_score",
    }


def _score_exact(row: dict[str, Any], dataset: str, format_score: float) -> dict[str, float]:
    output = row.get("output", row.get("response", row.get("solution_str", "")))
    ground_truth = row.get("gts", row.get("ground_truth", row.get("gt", "")))
    extra_info = dict(row)

    if dataset == "virl39k":
        result = virl39k.compute_score(
            str(output),
            str(ground_truth),
            extra_info=extra_info,
            format_score=format_score,
        )
    elif dataset == "vlaa":
        result = vlaa.compute_score(
            str(output),
            str(ground_truth),
            extra_info=extra_info,
            format_score=format_score,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    return {
        "acc": float(result.get("acc", 0.0)),
        "score": float(result.get("score", 0.0)),
        "format": float(result.get("format", 0.0)),
        "source": "exact",
    }


def _score_row(
    row: dict[str, Any],
    dataset: str,
    format_score: float,
    mode: str,
) -> dict[str, float]:
    if mode in {"auto", "logged"}:
        scored = _score_from_logged(row, format_score)
        if scored is not None:
            return scored
        if mode == "logged":
            raise ValueError("row has no score/reward/acc field for --mode logged")
    return _score_exact(row, dataset, format_score)


def _jsonl_files_for_run(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"), key=lambda p: (_step_from_file(p, 0), p.name))


def _discover_run_dirs(inputs: list[str]) -> list[Path]:
    candidates: list[Path] = []
    raw_inputs = inputs or DEFAULT_ROOTS

    for raw in raw_inputs:
        matches = glob.glob(raw)
        if not matches:
            matches = [raw]
        for match in matches:
            path = Path(match).expanduser()
            if path.is_file() and path.suffix == ".jsonl":
                candidates.append(path)
            elif path.is_dir() and list(path.glob("*.jsonl")):
                candidates.append(path)
            elif path.is_dir():
                candidates.extend(p for p in path.rglob("*_val") if p.is_dir() and list(p.glob("*.jsonl")))

    # De-duplicate while preserving sorted stability.
    unique = sorted({p.resolve() for p in candidates}, key=lambda p: str(p))
    return unique


def _run_name(path: Path) -> str:
    if path.is_file():
        return path.parent.name
    return path.name.removesuffix("_val")


def rescore_run(
    run_path: Path,
    dataset_arg: str,
    format_score: float,
    mode: str,
    rows_writer: csv.DictWriter | None = None,
) -> list[dict[str, Any]]:
    files = _jsonl_files_for_run(run_path)
    out_rows: list[dict[str, Any]] = []
    if not files:
        return out_rows

    run_name = _run_name(run_path)
    for ordinal, file_path in enumerate(files):
        raw_rows = list(_read_jsonl(file_path))
        if not raw_rows:
            continue

        dataset = _infer_dataset(run_path, raw_rows, dataset_arg)
        fixed = [_score_row(row, dataset, format_score, mode) for row in raw_rows]
        step_values = [_as_float(row.get("step")) for row in raw_rows]
        step = int(_mean(step_values) or _step_from_file(file_path, ordinal))

        old_acc = [_as_float(row.get("acc")) for row in raw_rows]
        old_score = [_as_float(row.get("score")) for row in raw_rows]
        old_reward = [_as_float(row.get("reward")) for row in raw_rows]

        acc = _mean(item["acc"] for item in fixed)
        score = _mean(item["score"] for item in fixed)
        fmt = _mean(item["format"] for item in fixed)
        n_logged = sum(1 for item in fixed if item["source"] == "logged_score")
        n_exact = sum(1 for item in fixed if item["source"] == "exact")

        out_rows.append(
            {
                "run": run_name,
                "dataset": dataset,
                "step": step,
                "n": len(raw_rows),
                "acc": acc,
                "score": score,
                "format": fmt,
                "logged_acc": _mean(old_acc),
                "logged_score": _mean(old_score),
                "logged_reward": _mean(old_reward),
                "n_from_logged_score": n_logged,
                "n_exact_rescore": n_exact,
                "path": str(file_path),
            }
        )

        if rows_writer is not None:
            for row, scored in zip(raw_rows, fixed):
                rows_writer.writerow(
                    {
                        "run": run_name,
                        "dataset": dataset,
                        "step": step,
                        "uid": row.get("uid", ""),
                        "index": row.get("index", ""),
                        "acc": scored["acc"],
                        "score": scored["score"],
                        "format": scored["format"],
                        "logged_acc": _as_float(row.get("acc")),
                        "logged_score": _as_float(row.get("score")),
                        "logged_reward": _as_float(row.get("reward")),
                        "source": scored["source"],
                    }
                )

    out_rows.sort(key=lambda row: int(row["step"]))
    return out_rows


def write_plot(curve_rows: list[dict[str, Any]], plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is not installed; rerun without --plot") from exc

    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in curve_rows:
        by_run.setdefault(str(row["run"]), []).append(row)

    plt.figure(figsize=(10, 6))
    for run, rows in sorted(by_run.items()):
        rows = sorted(rows, key=lambda item: int(item["step"]))
        steps = [int(item["step"]) for item in rows]
        accs = [float(item["acc"]) for item in rows]
        plt.plot(steps, accs, marker="o", linewidth=1.5, markersize=3, label=run)

    plt.xlabel("training step")
    plt.ylabel("true validation accuracy")
    plt.grid(True, alpha=0.25)
    if len(by_run) <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score validation JSONL dumps and write true accuracy curves."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Validation directories/files/globs. If a directory does not directly "
            "contain JSONL files, all nested *_val directories are processed. "
            "Defaults to the known peyman output roots."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", "virl39k", "vlaa"],
        default="auto",
        help="Reward scorer to use. auto uses virl39k for paths containing virl39k, else vlaa.",
    )
    parser.add_argument(
        "--format-score",
        type=float,
        default=0.1,
        help="Format reward weight used by the original scorer.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "logged", "exact"],
        default="auto",
        help=(
            "auto recovers acc from saved shaped score when possible and falls "
            "back to exact scorer; logged requires saved score/reward/acc; exact "
            "reruns the reward scorer for every row."
        ),
    )
    parser.add_argument(
        "--out",
        default="validation_accuracy_curves.csv",
        help="Output CSV path for one row per run/step.",
    )
    parser.add_argument(
        "--rows-out",
        default=None,
        help="Optional per-example CSV path with recomputed acc/score/format.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional PNG path for a combined accuracy plot.",
    )
    args = parser.parse_args()

    run_paths = _discover_run_dirs(args.paths)
    if not run_paths:
        print("No validation JSONL files found.", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run",
        "dataset",
        "step",
        "n",
        "acc",
        "score",
        "format",
        "logged_acc",
        "logged_score",
        "logged_reward",
        "n_from_logged_score",
        "n_exact_rescore",
        "path",
    ]

    row_fh = None
    rows_writer = None
    if args.rows_out:
        rows_path = Path(args.rows_out)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        row_fh = rows_path.open("w", newline="", encoding="utf-8")
        rows_writer = csv.DictWriter(
            row_fh,
            fieldnames=[
                "run",
                "dataset",
                "step",
                "uid",
                "index",
                "acc",
                "score",
                "format",
                "logged_acc",
                "logged_score",
                "logged_reward",
                "source",
            ],
        )
        rows_writer.writeheader()

    all_rows: list[dict[str, Any]] = []
    try:
        for run_path in run_paths:
            curve = rescore_run(
                run_path=run_path,
                dataset_arg=args.dataset,
                format_score=args.format_score,
                mode=args.mode,
                rows_writer=rows_writer,
            )
            all_rows.extend(curve)
            if curve:
                print(
                    f"{_run_name(run_path)}: {len(curve)} steps, "
                    f"final acc={curve[-1]['acc']:.6f} at step {curve[-1]['step']}"
                )
    finally:
        if row_fh is not None:
            row_fh.close()

    all_rows.sort(key=lambda row: (str(row["run"]), int(row["step"])))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} curve rows to {out_path}")

    if args.plot:
        plot_path = Path(args.plot)
        write_plot(all_rows, plot_path)
        print(f"Wrote plot to {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
