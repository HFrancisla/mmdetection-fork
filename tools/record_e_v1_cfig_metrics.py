#!/usr/bin/env python3
"""Append E-V1 C/F/G/I test results and three-seed means to FrontNet metrics."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


LABELS = {
    "c1_reflectance_fdsp": "C1 · Reflectance proxy + FDSP",
    "c2_iim_fdsp_interaction": "C2 · I→F interaction",
    "c3_reflectance_interaction": "C3 · Reflectance + interaction",
    "f2_dwt_fdsp": "F2 · DWT-FDSP",
    "f3_single_dw3": "F3 · Single DW3",
    "f3_multiscale_fdsp": "F3 · Multi-scale DW3×DW9",
    "f4_adaptive_fdsp": "F4 · Adaptive FDSP",
    "g1_mean": "G1 · Mean fusion",
    "i0_cnn_control": "I0 · CNN control",
    "i1_color_ratio": "I1 · Color ratio",
    "i2_unconstrained": "I2 · Unconstrained IIM",
    "i3_logrgb_filter": "I3 · Log-RGB filter",
}
SEEDS = ("1", "42", "2026")


def replace_section(path: Path, marker: str, section: str) -> None:
    text = path.read_text()
    if marker in text:
        text = text[: text.index(marker)].rstrip()
    path.write_text(text.rstrip() + "\n\n" + section.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_dir", type=Path, help="C/F/G/I test output directory")
    parser.add_argument(
        "--frontnet-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "FrontNet",
        help="FrontNet results directory",
    )
    args = parser.parse_args()

    test_dir = args.test_dir.resolve()
    if (test_dir / "overall.status").read_text().strip() != "SUCCESS":
        raise SystemExit("test queue is not marked SUCCESS")
    with (test_dir / "test_status.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 36 or any(row["status"] != "SUCCESS" for row in rows):
        raise SystemExit(f"expected 36 successful test rows, got {len(rows)}")

    grouped = defaultdict(list)
    for row in rows:
        variant = row["variant"]
        if variant not in LABELS:
            raise SystemExit(f"unexpected variant: {variant}")
        if row["seed"] not in SEEDS:
            raise SystemExit(f"unexpected seed for {variant}: {row['seed']}")
        grouped[variant].append(row)
    if set(grouped) != set(LABELS):
        raise SystemExit(f"variant set mismatch: {sorted(grouped)}")
    for variant, variant_rows in grouped.items():
        if len(variant_rows) != 3 or {row["seed"] for row in variant_rows} != set(SEEDS):
            raise SystemExit(f"expected seeds {SEEDS} exactly once for {variant}")

    def mean(variant: str, field: str) -> str:
        return f"{statistics.mean(float(row[field]) for row in grouped[variant]):.4f}"

    run_header = (
        "| 组 | 网络结构 | 配置文件 | seed | runs | test mAP@0.5 | mean Recall@0.5 | run |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"
    )
    run_rows = []
    for row in rows:
        variant = row["variant"]
        run_name = f"{variant}_seed{row['seed']}"
        run_rows.append(
            f"| {row['group']} | {LABELS[variant]} | `{row['config']}` | {row['seed']} | 1 "
            f"| {float(row['test_mAP']):.4f} | {float(row['test_macro_recall@0.5']):.4f} | `{run_name}` |"
        )
    run_section = "\n".join(
        ["### E-V1 Hybrid C/F/G/I 消融", "", run_header, *run_rows]
    )
    replace_section(
        args.frontnet_dir / "2-metrics/Hybrid/metrics.md",
        "### E-V1 Hybrid C/F/G/I 消融",
        run_section,
    )

    mean_header = (
        "| 组 | 网络结构 | 配置文件 | seed | runs | mAP@0.5 | mean Recall@0.5 |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: |"
    )
    mean_rows = []
    for variant, label in LABELS.items():
        first = grouped[variant][0]
        mean_rows.append(
            f"| {first['group']} | {label} | `{first['config']}` | 1, 42, 2026 | 3 "
            f"| {mean(variant, 'test_mAP')} | {mean(variant, 'test_macro_recall@0.5')} |"
        )
    mean_section = "\n".join(
        ["### E-V1 Hybrid C/F/G/I 消融", "", mean_header, *mean_rows]
    )
    replace_section(
        args.frontnet_dir / "metrics_hybrid.md",
        "### E-V1 Hybrid C/F/G/I 消融",
        mean_section,
    )

    f2 = grouped["f2_dwt_fdsp"][0]
    dwt_section = "\n".join(
        [
            "### E-V1 Hybrid DWT-FDSP",
            "",
            "| 方案 | 配置组合 | fdsp_type | seed | runs | residual | atan | mAP@0.5 | Recall@0.5 |",
            "| --- | --- | :---: | --- | ---: | :---: | :---: | ---: | ---: |",
            f"| E-V1 Hybrid | F2 · DWT-FDSP · `{f2['config']}` | dwt_fdsp | 1, 42, 2026 | 3 | — | — "
            f"| {mean('f2_dwt_fdsp', 'test_mAP')} | {mean('f2_dwt_fdsp', 'test_macro_recall@0.5')} |",
        ]
    )
    replace_section(
        args.frontnet_dir / "metrics_dwt.md",
        "### E-V1 Hybrid DWT-FDSP",
        dwt_section,
    )
    print(f"Recorded 36 runs and 12 three-seed means under {args.frontnet_dir}")


if __name__ == "__main__":
    main()
