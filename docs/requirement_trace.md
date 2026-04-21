# 需求落地清单

## 对照 `需求.md`

### 1. 需要有传统算法参照对象，便于对比提升

已实现：

- 传统基线：`src/recommender.py`
  - 基于近邻协同的基线推荐
- 优化模型：`src/rf_ranker.py`
  - 随机森林二阶段优化
- 输出：
  - `outputs/metrics/model_metrics.json`
  - `outputs/figures/fig_04_metric_comparison.png`
  - `outputs/figures/step_03_metrics.png`

### 2. 给关键步骤截图，并生成一系列关键可视化图像

已实现：

- 关键步骤图
  - `outputs/figures/step_01_overview.png`
  - `outputs/figures/step_02_clusters.png`
  - `outputs/figures/step_03_metrics.png`
- 主要可视化
  - `outputs/figures/fig_01_k_selection.png`
  - `outputs/figures/fig_02_cluster_scatter.png`
  - `outputs/figures/fig_03_cluster_profile.png`
  - `outputs/figures/fig_04_metric_comparison.png`
  - `outputs/figures/fig_05_feature_importance.png`
  - `outputs/figures/fig_06_correlation_heatmap.png`

### 3. `test2.csv` 是数据结构，需要找到网上真实数据集并标注来源

已实现：

- 数据抓取脚本：
  - `scripts/fetch_public_data_samples.py`
- 结构化大样本脚本：
  - `scripts/build_amazon_structured_dataset.py`
- 已生成大样本：
  - `data/amazon_fashion_structured_30k.csv`
  - 当前 35173 行
- 数据来源说明：
  - `docs/public_data_sources.md`
- 来源清单与样本：
  - `data/external/source_manifest.json`
  - `data/external/amazon_fashion/*`
  - `data/external/hm_sample/*`

### 4. 标签和输出图片中的英文需要替换成中文

已实现：

- 图表标题中文化
- 坐标轴中文化
- 图例中文化
- 表头中文化
- 指标名中文化
- 特征重要性中文化

相关文件：

- `src/visualizer.py`
- `src/output_labels.py`
- `scripts/run_pipeline.py`

## 对照开题报告

### 1. 多源数据采集与预处理

- `scripts/fetch_public_data_samples.py`
- `scripts/build_amazon_structured_dataset.py`
- `src/data_loader.py`

### 2. KMeans 用户分群

- `src/cluster_model.py`
- `outputs/tables/cluster_summary.csv`

### 3. 逻辑回归自动标注

- `src/cluster_model.py`
- 结果写入 `outputs/metrics/model_metrics.json`

### 4. 协同过滤推荐模型

- `src/recommender.py`

### 5. 随机森林优化推荐效果

- `src/rf_ranker.py`
- `outputs/tables/feature_importance.csv`

### 6. 指标评估

- 准确率
- 精确率
- 召回率
- F1 值
- 曲线下面积

### 7. 真实代码、结果、图表、日志

- 真实代码：`src/*.py`, `scripts/*.py`
- 运行日志：`outputs/run_log.txt`
- 图表：`outputs/figures/*.png`
- 报告页：`outputs/report.html`
