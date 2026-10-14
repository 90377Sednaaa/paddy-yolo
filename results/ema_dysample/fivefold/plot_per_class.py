"""Extract per-class held-out test scores from Kaggle logs and render a PNG."""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).resolve().parent
    log = (directory / "logs/paper_training.log").read_text(encoding="utf-8")
    log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
    destinations = list(re.finditer(
        r"Results saved to [^\n]*fold(\d+)/evaluate/test", log
    ))
    assert len(destinations) == 5
    class_pattern = re.compile(
        r"^\s*(classID_\d{2})\s+(\d+)\s+(\d+)\s+"
        r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    )
    names = [f"classID_{index:02d}" for index in range(16)]
    values = np.zeros((5, 16, 5))
    instances = np.zeros((5, 16), dtype=int)
    parsed_folds = set()
    rows = []
    for destination in destinations:
        fold = int(destination.group(1))
        assert fold in range(5) and fold not in parsed_folds
        parsed_folds.add(fold)
        # The final 16 class rows preceding this marker are its test table;
        # earlier rows belong to validation or the training final evaluation.
        table_rows = class_pattern.findall(log[max(0, destination.start() - 12000):destination.start()])[-16:]
        assert [row[0] for row in table_rows] == names
        for index, (name, images, boxes, precision, recall, ap50, ap5095) in enumerate(table_rows):
            precision, recall, ap50, ap5095 = map(float, (precision, recall, ap50, ap5095))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            values[fold, index] = [precision, recall, f1, ap50, ap5095]
            instances[fold, index] = int(boxes)
            rows.append([fold, name, int(images), int(boxes), precision, recall, f1, ap50, ap5095])
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    manifest = json.loads((directory / "splits/fold_manifest.json").read_text(encoding="utf-8"))
    expected_instances = np.sum([record["class_boxes"] for record in manifest["records"]], axis=0)
    np.testing.assert_array_equal(instances.sum(axis=0), expected_instances)
    assert instances.sum() == manifest["instances"] == 2281
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for report in summary["folds"]:
        fold = report["fold"]
        # Class console values are rounded to three significant figures.
        for column, key in [(0, "precision"), (1, "recall"), (3, "mAP50"), (4, "mAP50_95")]:
            assert abs(values[fold, :, column].mean() - report["metrics"]["test"][key]) < 0.00051

    with (directory / "per_class_fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "class", "images_with_class", "instances", "precision", "recall", "f1_derived", "AP50", "AP50_95"])
        writer.writerows(rows)
    means = values.mean(axis=0)
    counts = instances.sum(axis=0)
    with (directory / "per_class_mean_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "total_test_instances", "precision_mean", "recall_mean", "f1_derived_mean", "AP50_mean", "AP50_95_mean"])
        writer.writerows([[name, int(count), *row] for name, count, row in zip(names, counts, means)])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    figure = plt.figure(figsize=(17, 11), facecolor="white")
    grid = figure.add_gridspec(1, 2, left=0.185, right=0.96, top=0.775, bottom=0.205, wspace=0.085)
    panels = [means * 100, values[:, :, 3].T * 100]
    headings = ["Five-fold mean (%)", "AP50 by test fold (%)"]
    columns = [["Precision", "Recall", "F1*", "AP50", "AP50–95"], [f"Fold {index + 1}" for index in range(5)]]
    for panel, (data, heading, labels) in enumerate(zip(panels, headings, columns)):
        axis = figure.add_subplot(grid[panel])
        heatmap = axis.imshow(data, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        axis.set_title(heading, loc="left", pad=46, fontsize=15, weight="bold", color="#172E49")
        axis.set_xticks(np.arange(5), labels, fontsize=11)
        axis.xaxis.tick_top()
        axis.set_yticks(np.arange(16))
        axis.set_yticklabels([f"{name}  (n={count})" for name, count in zip(names, counts)] if panel == 0 else [])
        axis.tick_params(axis="both", length=0, pad=9, colors="#344054")
        axis.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 16, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.6)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for (row, column), score in np.ndenumerate(data):
            axis.text(column, row, f"{score:.1f}", ha="center", va="center", fontsize=11.5,
                      color="white" if score >= 61 else "#172E49")

    color_axis = figure.add_axes([0.69, 0.15, 0.27, 0.015])
    colorbar = figure.colorbar(heatmap, cax=color_axis, orientation="horizontal", ticks=[0, 20, 40, 60, 80, 100])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=10, colors="#667085")
    figure.text(0.06, 0.945, "EMA + DySample", fontsize=26, weight="bold", color="#172E49")
    figure.text(0.06, 0.902, "Per-class performance across five test folds", fontsize=21, color="#344054")
    figure.text(0.06, 0.86, "R2000  |  16 classes  |  3:1:1 rotation  |  300 epochs per fold", fontsize=12, color="#667085")
    figure.text(0.06, 0.164, "n = total test instances across five folds; each image is tested once.", fontsize=10.5, color="#667085")
    figure.text(0.06, 0.119, "Means are unweighted averages of five held-out test scores, using each fold’s best validation checkpoint.", fontsize=10.5, color="#667085")
    figure.text(0.06, 0.092, "* F1 = 2PR / (P + R), computed per class per fold, then averaged. Values are approximate: source logs use rounded scores.", fontsize=10.5, color="#667085")
    figure.text(0.06, 0.065, "Class IDs follow the dataset labels. Fold 1–5 = directories fold0–fold4. Source: logs/paper_training.log.", fontsize=10.5, color="#667085")
    output = directory / "per_class_performance.png"
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)
    print(f"Saved {output}")
    print("Verified: 5 test folds, 16 classes each, 2,281 test instances, and rounded macro metrics match summary.json.")


if __name__ == "__main__":
    main()
