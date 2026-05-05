import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_RUNS_DIR = ROOT_DIR / "runs" / "yolo"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reports" / "task3"
SCORE_COLUMN = "metrics/mAP50-95(B)"
RUNS_DIR = DEFAULT_RUNS_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def find_completed_runs(runs_dir):
    run_dirs = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "run_summary.json").exists() and (child / "results.csv").exists():
            run_dirs.append(child)
    return run_dirs


def read_run_record(run_dir):
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    results_df = pd.read_csv(run_dir / "results.csv")
    if results_df.empty:
        return None, None

    score_column = SCORE_COLUMN if SCORE_COLUMN in results_df.columns else "metrics/mAP50(B)"
    best_index = results_df[score_column].idxmax()
    best_row = results_df.loc[best_index].to_dict()

    record = {
        "run_name": run_dir.name,
        "preset": summary.get("preset"),
        "data_yaml": summary.get("data_yaml"),
        "model": summary.get("model"),
        "epochs_configured": summary.get("epochs"),
        "batch": summary.get("batch"),
        "imgsz": summary.get("imgsz"),
        "device": summary.get("device"),
        "fraction": summary.get("fraction"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "best_checkpoint": summary.get("best_checkpoint"),
        "last_checkpoint": summary.get("last_checkpoint"),
        "best_epoch": int(best_row.get("epoch", 0)),
        "best_score_column": score_column,
    }
    record.update(best_row)
    return record, results_df


def copy_run_artifacts(run_dir, output_run_dir):
    output_run_dir.mkdir(parents=True, exist_ok=True)
    for file_name in [
        "results.csv",
        "results.png",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "run_summary.json",
    ]:
        source = run_dir / file_name
        if source.exists():
            shutil.copy2(source, output_run_dir / file_name)


def save_comparison_plot(summary_df, output_dir, metric_column, file_name, title):
    if metric_column not in summary_df.columns:
        return

    plot_df = summary_df[["run_name", metric_column]].dropna().sort_values(by=metric_column, ascending=False)
    if plot_df.empty:
        return

    ax = plot_df.plot(kind="bar", x="run_name", y=metric_column, figsize=(10, 6), legend=False, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel(metric_column)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / file_name, dpi=300)
    plt.close()


def main():
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu runów: {RUNS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_runs_dir = OUTPUT_DIR / "runs"
    report_runs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for run_dir in find_completed_runs(RUNS_DIR):
        run_record, _ = read_run_record(run_dir)
        if run_record is None:
            continue
        records.append(run_record)
        copy_run_artifacts(run_dir, report_runs_dir / run_dir.name)

    if not records:
        raise SystemExit("Nie znaleziono żadnych zakończonych runów z run_summary.json i results.csv.")

    summary_df = pd.DataFrame(records).sort_values(by=["elapsed_seconds", "run_name"], na_position="last")
    summary_df.to_csv(OUTPUT_DIR / "task3_runs_summary.csv", index=False)

    save_comparison_plot(summary_df, OUTPUT_DIR, "metrics/mAP50(B)", "comparison_map50.png", "Task 3 comparison: mAP50")
    save_comparison_plot(summary_df, OUTPUT_DIR, SCORE_COLUMN, "comparison_map50_95.png", "Task 3 comparison: mAP50-95")
    save_comparison_plot(summary_df, OUTPUT_DIR, "elapsed_seconds", "comparison_runtime.png", "Task 3 comparison: runtime")

    print(f"Zebrano {len(summary_df)} runów do {OUTPUT_DIR}")


if __name__ == "__main__":
    main()