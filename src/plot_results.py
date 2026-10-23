"""Regenerate report figures from the recorded experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
LABELS = ["NONE", "，", "。", "？", "！", "；", "：", "、"]
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 220,
    }
)


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def plot_training() -> None:
    data = read_json("training_metrics.json")
    epochs = [item["epoch"] for item in data["epochs"]]
    loss = [item["train_loss"] for item in data["epochs"]]
    validation_f1 = [item["validation"]["punctuation_micro_f1"] for item in data["epochs"]]
    fig, (ax_loss, ax_f1) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
    ax_loss.plot(epochs, loss, marker="o", linewidth=2, color="#4C78A8")
    ax_loss.set_ylabel("训练损失")
    ax_loss.set_title("训练收敛与验证集指标")
    ax_loss.grid(axis="y", alpha=0.25)
    ax_f1.plot(epochs, validation_f1, marker="o", linewidth=2, color="#F58518")
    ax_f1.set_ylabel("标点 micro-F1")
    ax_f1.set_xlabel("训练轮次")
    ax_f1.set_xticks(epochs)
    ax_f1.set_ylim(min(validation_f1) - 0.015, max(validation_f1) + 0.015)
    ax_f1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_validation.png", bbox_inches="tight")
    plt.close(fig)


def plot_test_classes() -> None:
    data = read_json("test_metrics.json")["metrics"]["per_class"]
    punctuation_names = {
        "，": "逗号",
        "。": "句号",
        "？": "问号",
        "！": "叹号",
        "；": "分号",
        "：": "冒号",
        "、": "顿号",
    }
    labels = [
        f"{punctuation_names[label]}（n={data[label]['support']:,}）"
        for label in data
    ]
    f1 = [data[label]["f1"] for label in data]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    bars = ax.barh(y, f1, color="#54A24B", alpha=0.92)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1")
    ax.set_title("测试集标点类别 F1（括号内为真实支持数）")
    ax.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, f1):
        ax.text(
            value + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "test_per_class.png", bbox_inches="tight")
    plt.close(fig)


def plot_test_confusion() -> None:
    with (RESULTS / "test_confusion_matrix.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.reader(handle))
    matrix = np.asarray([[int(value) for value in row[1:]] for row in rows[1:]], dtype=np.int64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_totals, 1)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(LABELS)), LABELS)
    ax.set_yticks(np.arange(len(LABELS)), LABELS)
    ax.set_xlabel("模型预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("测试集逐字混淆矩阵（按真实类别归一化）")
    threshold = 0.48
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            fraction = normalized[i, j]
            ax.text(
                j,
                i,
                f"{fraction:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if fraction > threshold else "#222222",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("真实类别内的比例")
    fig.tight_layout()
    fig.savefig(FIGURES / "test_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_long_runtime() -> None:
    data = read_json("long_runtime_benchmark.json")
    rows = [
        row
        for row in data["runtime_rows"]
        if row["method"].startswith("fixed_e2e_") or row["method"] == "original_serial_e2e"
    ]
    labels = []
    medians = []
    lower_errors = []
    upper_errors = []
    label_positions = []
    colors = []
    for row in rows:
        method = row["method"]
        times = [float(value) for value in row["raw_repetition_seconds"]]
        unique_chars = int(data["test_long_unique_character_positions"])
        rates = [unique_chars / value for value in times]
        median_rate = float(row["chars_per_second"])
        if method.startswith("fixed_e2e_"):
            batch = method.split("bs", 1)[1].split("_", 1)[0]
            labels.append(f"固定窗 batch {batch}")
            colors.append("#F58518")
        else:
            labels.append("原串行自适应窗口")
            colors.append("#4C78A8")
        medians.append(median_rate)
        lower_errors.append(median_rate - min(rates))
        upper_errors.append(max(rates) - median_rate)
        label_positions.append(max(rates) * 1.02)

    y = np.arange(len(medians))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bars = ax.barh(
        y,
        medians,
        color=colors,
        xerr=np.asarray([lower_errors, upper_errors]),
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#444444"},
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("唯一字符位置 / 秒（中位数；误差线为三次实测范围）")
    ax.set_title("长文测试子集端到端吞吐率")
    ax.set_xlim(0, max(label_positions) * 1.08)
    ax.grid(axis="x", alpha=0.22)
    for bar, value, label_x in zip(bars, medians, label_positions):
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            ha="left",
            color="#222222",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "long_text_runtime.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_training()
    plot_test_classes()
    plot_test_confusion()
    plot_long_runtime()
    print(
        "Wrote training_validation.png, test_per_class.png, "
        "test_confusion_matrix.png, and long_text_runtime.png"
    )


if __name__ == "__main__":
    main()
