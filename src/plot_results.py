"""Regenerate the compact report figures from recorded experiment JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
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
        "，": "逗号（，）",
        "。": "句号（。）",
        "？": "问号（？）",
        "！": "叹号（！）",
        "；": "分号（；）",
        "：": "冒号（：）",
        "、": "顿号（、）",
    }
    labels = [punctuation_names[label] for label in data]
    f1 = [data[label]["f1"] for label in data]
    support = [data[label]["support"] for label in data]
    y = np.arange(len(labels))
    fig, ax_f1 = plt.subplots(figsize=(7.8, 4.7))
    bars = ax_f1.barh(y, f1, color="#54A24B", alpha=0.9)
    ax_f1.set_yticks(y, labels)
    ax_f1.invert_yaxis()
    ax_f1.set_xlim(0, 1.05)
    ax_f1.set_xlabel("F1")
    ax_f1.set_title("测试集各标点类别 F1 与真实支持数")
    ax_f1.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, f1):
        ax_f1.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    ax_support = ax_f1.twiny()
    ax_support.plot(support, y, marker="|", markersize=11, linewidth=0, color="#4C78A8")
    ax_support.set_xlabel("真实支持数（蓝色刻度）")
    ax_support.set_xlim(0, max(support) * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_per_class.png", bbox_inches="tight")
    plt.close(fig)


def plot_long_runtime() -> None:
    data = read_json("long_runtime_benchmark.json")
    rows = data["runtime_rows"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    labels = []
    values = []
    colors = []
    for row in rows:
        method = row["method"]
        if method.startswith("fixed_forward_"):
            batch = method.split("bs", 1)[1]
            labels.append(f"固定窗 forward B{batch}")
            colors.append("#4C78A8")
        elif method.startswith("fixed_e2e_"):
            batch = method.split("bs", 1)[1].split("_", 1)[0]
            labels.append(f"固定窗端到端 B{batch}")
            colors.append("#F58518")
        else:
            labels.append("原串行端到端")
            colors.append("#E45756")
        values.append(float(row["chars_per_second"]))
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("唯一字符位置 / 秒（中位数）")
    ax.set_title("长文测试子集吞吐率：模型前向与端到端流程")
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(value * 1.01, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "long_text_runtime.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_training()
    plot_test_classes()
    plot_long_runtime()
    print("Wrote training_validation.png, test_per_class.png, and long_text_runtime.png")


if __name__ == "__main__":
    main()
