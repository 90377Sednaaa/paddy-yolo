"""Render the downloaded five-fold test metrics and their arithmetic mean."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def main():
    directory = Path(__file__).resolve().parent
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    folds = sorted(summary["folds"], key=lambda item: item["fold"])
    assert summary["completed_folds"] == len(folds) == 5
    assert [fold["fold"] for fold in folds] == list(range(5))
    metrics = ["precision", "recall", "f1", "mAP50", "mAP50_95"]
    titles = ["Precision", "Recall", "F1", "mAP50", "mAP50–95"]
    values = np.array([[fold["metrics"]["test"][key] for key in metrics] for fold in folds])
    means = values.mean(axis=0)
    np.testing.assert_allclose(
        means,
        [summary["aggregate"]["test"][key]["mean"] for key in metrics],
        rtol=0,
        atol=1e-12,
    )
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    percentages = np.vstack([values, means]) * 100
    labels = [f"Fold {index + 1}" for index in range(5)] + ["Mean"]
    colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#AA3377", "#172E49"]

    log = (directory / "logs/paper_training.log").read_text(encoding="utf-8")
    speed_pattern = (
        r"Speed: ([0-9.]+)ms preprocess, ([0-9.]+)ms inference, "
        r"([0-9.]+)ms loss, ([0-9.]+)ms postprocess per image\s+"
        r"Results saved to [^\n]*fold(\d+)/evaluate/test"
    )
    speed_rows = re.findall(speed_pattern, log)
    assert len(speed_rows) == 5
    timings = {int(fold): (float(pre), float(inference), float(post))
               for pre, inference, loss, post, fold in speed_rows}
    assert sorted(timings) == list(range(5))
    inference_times = np.array([timings[index][1] for index in range(5)])
    mean_inference = inference_times.mean()
    mean_processing = np.mean([sum(timings[index]) for index in range(5)])
    reported_flops = {float(value) for value in re.findall(
        r"summary \(fused\):[^\n]*?([0-9.]+) GFLOPs", log
    )}
    assert len(reported_flops) == 1
    gflops = reported_flops.pop()
    efficiency = [
        [f"{gflops:.1f}", f"{time:.2f}", f"{1000 / time:.2f}"]
        for time in [*inference_times, mean_inference]
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    figure = plt.figure(figsize=(16, 11), facecolor="white")
    grid = figure.add_gridspec(
        2, 1, height_ratios=[1.65, 1], left=0.075, right=0.965,
        top=0.79, bottom=0.20, hspace=0.28,
    )
    axis = figure.add_subplot(grid[0])
    positions = np.arange(len(metrics))
    width = 0.115
    for index, (label, color) in enumerate(zip(labels, colors)):
        axis.bar(
            positions + (index - 2.5) * width, percentages[index],
            width=width * 0.91, label=label, color=color,
            hatch="//" if label == "Mean" else None,
            edgecolor="white", linewidth=0.7, zorder=3,
        )
    axis.set_xticks(positions, titles, fontsize=13)
    axis.set_ylim(0, 100)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_ylabel("Test score", labelpad=12)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E4E9EE", linewidth=0.8)
    axis.tick_params(axis="both", length=0, pad=10, colors="#344054")
    for name in ["top", "right", "left"]:
        axis.spines[name].set_visible(False)
    axis.spines["bottom"].set_color("#D4DCE5")
    axis.legend(
        ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.06),
        frameon=False, handlelength=1.5, columnspacing=2.8,
    )

    table_axis = figure.add_subplot(grid[1])
    table_axis.axis("off")
    table_data = [[label] + [f"{value:.2f}%" for value in row] + speed
                  for label, row, speed in zip(labels, percentages, efficiency)]
    table = table_axis.table(
        cellText=table_data, colLabels=["Result"] + titles + [
            "GFLOPs", "Inference\n(ms/image)", "FPS\n(inference)",
        ],
        cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1],
        colWidths=[0.12] + [0.112] * 5 + [0.10, 0.11, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)
        cell.set_height((1.65 if row == 0 else 1) / 7.65)
        if row == 0:
            cell.set_facecolor("#172E49")
            cell.set_text_props(color="white", weight="bold")
        elif row == 6:
            cell.set_facecolor("#DDE8F2")
            cell.set_text_props(color="#172E49", weight="bold")
        else:
            cell.set_facecolor("#F0F4F8" if row % 2 else "#F8FAFC")
            cell.set_text_props(color="#263547")
            if column == 0:
                cell.set_text_props(color=colors[row - 1], weight="bold")

    figure.text(0.075, 0.94, "EMA + DySample", fontsize=26, weight="bold", color="#172E49")
    figure.text(0.075, 0.895, "Five-fold test results and mean", fontsize=20, color="#344054")
    figure.text(
        0.075, 0.853,
        "R2000  |  3:1:1 train / validation / test rotation  |  300 epochs per fold",
        fontsize=12, color="#667085",
    )
    figure.text(
        0.075, 0.157,
        "Mean = arithmetic average of five test scores, calculated before rounding. "
        "Each fold uses its best validation checkpoint.",
        fontsize=10.5, color="#667085",
    )
    figure.text(
        0.075, 0.129,
        "Speed: Tesla T4, evaluation batch 32, imgsz 640. Batched throughput; not batch-1 latency. GFLOPs: Ultralytics estimate.",
        fontsize=10.5, color="#667085",
    )
    figure.text(
        0.075, 0.101,
        "FPS = 1000 / inference time; mean FPS uses mean inference time. Timings are approximate, derived from rounded logs.",
        fontsize=10.5, color="#667085",
    )
    figure.text(
        0.075, 0.073,
        f"Mean preprocessing + inference + postprocessing: {mean_processing:.2f} ms/image "
        f"({1000 / mean_processing:.2f} FPS). Excludes data loading and saving.",
        fontsize=10.5, color="#667085",
    )
    figure.text(
        0.075, 0.045,
        "Fold 1–5 = directories fold0–fold4. Sources: downloaded summary.json and logs/paper_training.log.",
        fontsize=10.5, color="#667085",
    )
    output = directory / "fivefold_results.png"
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)
    print(f"Saved {output}")
    print("Verified test means (%): " + ", ".join(
        f"{title}={value:.2f}" for title, value in zip(titles, means * 100)
    ))


if __name__ == "__main__":
    main()
