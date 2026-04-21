# 电商用户画像与个性化推荐

本目录已经按照 `开题报告.docx` 和 `需求.md` 补齐为可运行项目，当前支持两套数据流程：

1. `test2.csv` 小样本验证版
2. 基于 Amazon Fashion 真实公开数据生成的 `3万+` 结构化实验版

## 当前交付内容

### 1. 真实代码

- `src/data_loader.py`：数据校验、切分、候选集构造
- `src/cluster_model.py`：KMeans 用户分群与逻辑回归自动标注
- `src/recommender.py`：传统近邻协同基线
- `src/rf_ranker.py`：随机森林二阶段优化
- `src/visualizer.py`：图表、表格图、HTML 报告
- `src/output_labels.py`：中文输出标签映射

### 2. 真实公开数据来源

- `scripts/fetch_public_data_samples.py`
  - 下载 Amazon Fashion 官方公开文件
  - 抓取 H&M 公开样本
- `scripts/build_amazon_structured_dataset.py`
  - 把 Amazon Fashion 原始公开数据转换成符合 `test2.csv` 结构的 32 列数据
  - 当前已生成：
    `data/amazon_fashion_structured_30k.csv`

### 3. 主实验流程

- `scripts/run_pipeline.py`
  - 用户画像聚类
  - 传统基线建模
  - 随机森林优化
  - 指标评估
  - 中文图表与关键步骤图输出

## 推荐运行方式

### 先拉公开数据样本

```powershell
python scripts/fetch_public_data_samples.py
```

### 生成 3 万条以上结构化数据

```powershell
python scripts/build_amazon_structured_dataset.py --output data/amazon_fashion_structured_30k.csv --min-rows 30000
```

### 运行完整实验

```powershell
python scripts/run_pipeline.py --input data/amazon_fashion_structured_30k.csv --output-dir outputs
```

## 当前结果

### 数据量

- `data/amazon_fashion_structured_30k.csv`
  - 35173 行
  - 32 列
  - 34491 个用户
  - 6331 个商品

### 输出文件

- `outputs/report.html`
- `outputs/run_log.txt`
- `outputs/metrics/model_metrics.json`
- `outputs/tables/*.csv`
- `outputs/figures/*.png`

### 图表中文化

当前输出图表中的标题、坐标轴、图例、表头、特征名、指标名都已替换为中文。

## 相关文档

- `docs/public_data_sources.md`
- `docs/requirement_trace.md`

## 依赖

```powershell
pip install -r requirements.txt
```
