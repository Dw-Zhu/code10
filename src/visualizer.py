from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_matplotlib() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.sans-serif"] = preferred_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 150


def save_table_image(df: pd.DataFrame, title: str, output_path: str | Path) -> None:
    configure_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_height = max(2.5, 0.45 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_k_selection_plot(k_metrics: pd.DataFrame, output_path: str | Path) -> None:
    configure_matplotlib()
    output_path = Path(output_path)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(k_metrics["k"], k_metrics["silhouette"], marker="o", color="#1f77b4")
    ax1.set_xlabel("聚类数")
    ax1.set_ylabel("轮廓系数", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(k_metrics["k"], k_metrics["davies_bouldin"], marker="s", color="#d62728")
    ax2.set_ylabel("戴维森堡丁指数", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("聚类数选择")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_cluster_scatter(clustered_df: pd.DataFrame, output_path: str | Path) -> None:
    configure_matplotlib()
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id, group in clustered_df.groupby("cluster_id"):
        label = str(group["cluster_name"].iloc[0]) if "cluster_name" in group.columns else "群体{0}".format(cluster_id)
        ax.scatter(group["pca_x"], group["pca_y"], s=16, alpha=0.65, label=label)
    ax.set_title("用户画像聚类分布")
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_cluster_profile(cluster_summary: pd.DataFrame, output_path: str | Path) -> None:
    configure_matplotlib()
    output_path = Path(output_path)
    plot_df = cluster_summary.set_index("cluster_name")[
        ["purchase_freq", "total_spend", "purchase_intent", "purchase_ratio"]
    ]
    plot_df = plot_df.rename(
        columns={
            "purchase_freq": "购买频次",
            "total_spend": "累计消费额",
            "purchase_intent": "购买意向",
            "purchase_ratio": "正样本占比",
        }
    )
    normalized = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min()).replace(0, 1)
    ax = normalized.plot(kind="bar", figsize=(10, 5), colormap="tab20c")
    ax.set_title("用户群体画像对比")
    ax.set_ylabel("标准化值")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    ax.figure.savefig(output_path, bbox_inches="tight")
    plt.close(ax.figure)


def save_metric_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    configure_matplotlib()
    output_path = Path(output_path)
    pivot = metrics_df.set_index("metric")
    ax = pivot.plot(kind="bar", figsize=(9, 4.8), color=["#7f8c8d", "#2e86de"])
    ax.set_title("基线模型与优化模型指标对比")
    ax.set_ylabel("指标值")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    ax.figure.savefig(output_path, bbox_inches="tight")
    plt.close(ax.figure)


def save_feature_importance(importance_df: pd.DataFrame, output_path: str | Path, top_n: int = 12) -> None:
    configure_matplotlib()
    output_path = Path(output_path)
    plot_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#16a085")
    ax.set_title("随机森林特征重要性前 {0} 项".format(top_n))
    ax.set_xlabel("重要性")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    output_path: str | Path,
    display_labels: List[str] | None = None,
) -> None:
    configure_matplotlib()
    output_path = Path(output_path)
    corr = df[columns].corr().to_numpy()
    labels = display_labels or columns

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("关键变量相关性热力图")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_html_report(
    output_path: str | Path,
    run_date: str,
    overview: Dict[str, float],
    auto_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    recommendation_preview: pd.DataFrame,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def render_rows(mapping: Dict[str, float]) -> str:
        rendered = []
        for key, value in mapping.items():
            number = float(value) if isinstance(value, (int, float)) else 0.0
            rendered.append("<tr><td>{0}</td><td>{1:.4f}</td></tr>".format(key, number))
        return "".join(rendered)

    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>电商用户画像与推荐实验报告</title>
  <style>
    body {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; margin: 32px; color: #222; background: #f7f8fb; }}
    h1, h2 {{ color: #0f2741; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ background: white; border-radius: 14px; padding: 18px 20px; box-shadow: 0 8px 28px rgba(15,39,65,0.08); }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #dfe6ee; padding: 8px 10px; font-size: 14px; }}
    th {{ background: #edf3f9; }}
    .muted {{ color: #5b6777; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>电商用户画像与个性化推荐实验报告</h1>
  <p class="muted">生成时间：{run_date}</p>
  <div class="grid">
    <div class="card">
      <h2>数据概况</h2>
      <table>{overview_rows}</table>
    </div>
    <div class="card">
      <h2>自动标注效果</h2>
      <table>{auto_rows}</table>
    </div>
    <div class="card">
      <h2>传统基线</h2>
      <table>{baseline_rows}</table>
    </div>
    <div class="card">
      <h2>随机森林优化</h2>
      <table>{optimized_rows}</table>
    </div>
  </div>
  <div class="card" style="margin-top: 18px;">
    <h2>推荐示例</h2>
    {recommendation_table}
  </div>
</body>
</html>""".format(
        run_date=run_date,
        overview_rows=render_rows(overview),
        auto_rows=render_rows(auto_metrics),
        baseline_rows=render_rows(baseline_metrics),
        optimized_rows=render_rows(optimized_metrics),
        recommendation_table=recommendation_preview.to_html(index=False, justify="center"),
    )
    output_path.write_text(html, encoding="utf-8")
