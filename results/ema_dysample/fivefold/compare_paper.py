"""Compare published R2000 Table 8 with recovered five-fold test results."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader


def style_table(table, delta_rows=(), delta_columns=()):
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.3)
        if row == 0:
            cell.set_facecolor("#172E49")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F0F4F8" if row % 2 else "#F8FAFC")
            cell.set_text_props(color="#263547")
            if (row in delta_rows and column > 0) or column in delta_columns:
                value = float(cell.get_text().get_text().replace("%", "").replace(" pp", ""))
                cell.set_facecolor("#E2F2EB" if value >= 0 else "#FBE8E6")
                cell.set_text_props(color="#147354" if value >= 0 else "#AE3430", weight="bold")
            if column == 0:
                cell.set_text_props(weight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    reader = PdfReader(str(args.paper))
    page_index = next(index for index, page in enumerate(reader.pages)
                      if "Category-specific detection results on R2000" in page.extract_text())
    text = reader.pages[page_index].extract_text(extraction_mode="layout")
    table_text = text.split("Table 8", 1)[1].split("Table 9", 1)[0]
    published = {}
    for line in table_text.splitlines():
        match = re.match(r"^\s*(All|\d+)\s+", line)
        if not match:
            continue
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%", line)
        if len(numbers) == 10:
            published[match.group(1)] = list(map(float, numbers[5:]))
    assert set(published) == {"All", *(str(index) for index in range(16))}
    # Paper order: precision, recall, AP50, AP50:95, F1.
    paper_keys = ["precision", "recall", "mAP50", "mAP50_95", "f1"]
    source = {
        "paper": str(args.paper.resolve()), "doi": "10.1016/j.compag.2025.110777",
        "table": 8, "pdf_page": page_index + 1, "dataset": "R2000",
        "model": "Paddy-YOLO", "units": "percent",
        "metric_order": paper_keys, "rows": published,
    }
    (directory / "paper_r2000_table8.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "per_class_mean_metrics.csv").open(encoding="utf-8", newline="") as handle:
        local = list(csv.DictReader(handle))
    assert [row["class"] for row in local] == [f"classID_{index:02d}" for index in range(16)]
    order = [0, 1, 4, 2, 3]
    keys = [paper_keys[index] for index in order]
    headings = ["Precision", "Recall", "F1", "mAP50", "mAP50–95"]
    paper_overall = np.array([published["All"][index] for index in order])
    enhanced_overall = np.array([summary["aggregate"]["test"][key]["mean"] * 100 for key in keys])
    deltas = enhanced_overall - paper_overall
    class_rows = []
    detailed_rows = []
    for index, row in enumerate(local):
        baseline = published[str(index)]
        ap50, ap95 = float(row["AP50_mean"]) * 100, float(row["AP50_95_mean"]) * 100
        class_rows.append([
            row["class"], f"{baseline[2]:.1f}%", f"{ap50:.2f}%", f"{ap50 - baseline[2]:+.2f}",
            f"{baseline[3]:.1f}%", f"{ap95:.2f}%", f"{ap95 - baseline[3]:+.2f}",
        ])
        for key, csv_key, paper_index in [
            ("precision", "precision_mean", 0), ("recall", "recall_mean", 1),
            ("f1", "f1_derived_mean", 4), ("AP50", "AP50_mean", 2), ("AP50_95", "AP50_95_mean", 3),
        ]:
            enhanced = float(row[csv_key]) * 100
            detailed_rows.append([row["class"], key, baseline[paper_index], enhanced, enhanced - baseline[paper_index]])
    with (directory / "paper_per_class_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "metric", "paper_percent", "enhanced_percent", "difference_percentage_points"])
        writer.writerows(detailed_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    figure = plt.figure(figsize=(16, 14), facecolor="white")
    figure.text(0.06, 0.95, "Comparison with the baseline paper", fontsize=25, weight="bold", color="#172E49")
    figure.text(0.06, 0.918, "Published Paddy-YOLO vs EMA + DySample  |  R2000", fontsize=18, color="#344054")
    figure.text(0.06, 0.875, "Overall test performance", fontsize=15, weight="bold", color="#172E49")
    overall_axis = figure.add_axes([0.06, 0.765, 0.90, 0.095])
    overall_axis.axis("off")
    overall_data = [
        ["Paper: Paddy-YOLO"] + [f"{value:.1f}%" for value in paper_overall],
        ["EMA + DySample"] + [f"{value:.2f}%" for value in enhanced_overall],
        ["Difference (pp)"] + [f"{value:+.2f}" for value in deltas],
    ]
    overall_table = overall_axis.table(
        cellText=overall_data, colLabels=["Model / comparison"] + headings,
        cellLoc="center", colLoc="center", colWidths=[0.24] + [0.152] * 5, bbox=[0, 0, 1, 1],
    )
    style_table(overall_table, delta_rows=(3,))
    figure.text(0.06, 0.724, "Per-class AP50 and AP50–95", fontsize=15, weight="bold", color="#172E49")
    class_axis = figure.add_axes([0.06, 0.195, 0.90, 0.51])
    class_axis.axis("off")
    class_table = class_axis.table(
        cellText=class_rows, colLabels=["Class ID", "Paper\nAP50", "Enhanced\nAP50", "Δ AP50\n(pp)",
                                           "Paper\nAP50–95", "Enhanced\nAP50–95", "Δ AP50–95\n(pp)"],
        cellLoc="center", colLoc="center", colWidths=[0.19] + [0.135] * 6, bbox=[0, 0, 1, 1],
    )
    style_table(class_table, delta_columns=(3, 6))
    for (row, column), cell in class_table.get_celld().items():
        cell.set_height((1.6 if row == 0 else 1) / 17.6)
    figure.text(0.06, 0.153, "pp = percentage points; positive differences indicate higher scores. Enhanced per-class scores use rounded test logs.", fontsize=10.5, color="#667085")
    figure.text(0.06, 0.127, "Published benchmark comparison: both use five-fold 3:1:1 evaluation, but the authors’ original fold memberships are unavailable.", fontsize=10.5, color="#667085")
    figure.text(0.06, 0.101, "Initial learning rate differs: paper 0.01, enhanced 0.001. These results do not isolate the contribution of the added modules.", fontsize=10.5, color="#667085")
    figure.text(0.06, 0.075, "Classes align by numeric ID. Class 4 has 118 vs 119 paper instances; class 7 has 66 vs 65. Other class totals match Table 4.", fontsize=10.5, color="#667085")
    figure.text(0.06, 0.049, "Sources: Zheng et al. (2025), Table 8 (PDF p. 8); local five-fold test summary and per-class logs. DOI: 10.1016/j.compag.2025.110777", fontsize=10.5, color="#667085")
    output = directory / "paper_baseline_comparison.png"
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)
    print(f"Saved {output}")
    print(dict(zip(headings, deltas.tolist())))
    print("Classes improved in AP50:", sum(float(row[3]) > 0 for row in class_rows), "/ 16")
    print("Classes improved in AP50–95:", sum(float(row[6]) > 0 for row in class_rows), "/ 16")


if __name__ == "__main__":
    main()
