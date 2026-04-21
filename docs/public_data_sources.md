# 真实公开数据来源与字段映射

## 已接入的公开来源

### 1. Amazon Fashion 官方公开文件

- 原始来源主页  
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon/
- 评分文件  
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/ratings_Amazon_Fashion.csv
- 商品元数据文件  
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Amazon_Fashion.json.gz

本项目已基于上述原始文件生成结构化实验数据：

- `data/amazon_fashion_structured_30k.csv`
- 当前样本量：35173 行

### 2. H&M Personalized Fashion Recommendations

- 官方竞赛页  
  https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
- 公开访问样本镜像页  
  https://huggingface.co/datasets/dinhlnd1610/HM-Personalized-Fashion-Recommendations

本项目使用 `scripts/fetch_public_data_samples.py` 抓取了客户、商品、交易样本，用于补充来源证明与字段参考：

- `data/external/hm_sample/customers_sample.csv`
- `data/external/hm_sample/articles_sample.csv`
- `data/external/hm_sample/transactions_sample.csv`

## Amazon 结构化数据的字段映射

### A. 直接来自公开原始数据的字段

- `user_id`
  - 来自评分文件用户编号
- `item_id`
  - 来自评分文件商品编号
- `price`
  - 来自商品元数据价格
- `title_length`
  - 来自商品标题长度
- `label`
  - 基于真实评分值构造，当前规则为 `rating >= 5`

### B. 基于真实日志聚合得到的字段

- `purchase_freq`
  - 用户历史购买次数
- `total_spend`
  - 用户历史累计消费额
- `register_days`
  - 用户首次与末次购买跨度
- `discount_rate`
  - 商品价格相对品类中位价的折扣代理
- `pv_count`
  - 商品交互量与查看关联量聚合
- `last_click_gap`
  - 用户相邻两次交互的时间间隔
- `interaction_rate`
  - 用户购买频次相对活跃周期的互动率
- `purchase_intent`
  - 用户历史正向反馈、商品评分与折扣的综合意向分
- `freshness_score`
  - 交互时间的新鲜度归一化得分
- `social_influence`
  - 商品交互热度与关联关系的综合影响力

### C. 代理特征

以下字段在 Amazon 原始文件中没有同名一手列，因此本项目明确按“代理特征”处理：

- `age`
  - 由用户活跃天数、累计消费额、平均价格分位映射得到
- `gender`
  - 由商品标题性别关键词聚合，缺失时使用稳定哈希回退
- `user_level`
  - 由消费额与购买频次分层得到
- `follow_num`
  - 由用户历史不同商品数近似
- `fans_num`
  - 由用户接触商品的平均热度近似
- `title_emo_score`
  - 由标题积极词命中率计算
- `img_count`
  - 由图片存在性和关联商品数量代理
- `has_video`
  - 由商品内容丰富度代理
- `like_num / comment_num / share_num / collect_num / is_follow_author / add2cart`
  - 由评分量、关联商品数量、商品热度和折扣信息代理

## 中文标签处理

为了满足“标签和输出图片用中文”的要求，本项目已将公开数据中的主要类别统一映射为中文：

- `Clothing` -> `服饰`
- `Shoes` -> `鞋靴`
- `Jewelry` -> `珠宝配饰`
- `Watches` -> `腕表`
- `Sports &amp; Outdoors` -> `运动户外`
- `Beauty` / `Health & Personal Care` -> `美妆个护`
- `Baby` -> `母婴用品`
- 其他尾部类别 -> `其他` 或 `家居杂货`

## 结论

1. 当前目录已经具备真实公开原始数据来源。
2. 当前目录已经具备 `3万+` 的结构化实验数据。
3. 当前实验图表和输出标签已经统一为中文。
