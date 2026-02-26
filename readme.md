# Divvy 共享单车需求预测

基于芝加哥 Divvy 共享单车数据与城市兴趣点（POI）数据，使用随机森林回归模型与滑动窗口时间序列验证，对各空间网格单元的单车行程需求进行预测。

---

## 项目概述

本项目包含两个核心功能：

1. **数据爬取与预处理**：从 Divvy Bikes 官网抓取站点数据，对原始行程记录进行清洗、格式转换与时间/空间聚合。
2. **时间序列需求预测**：将芝加哥城市 POI 数据与历史行程数据融合，基于 500 m × 500 m 空间网格，利用随机森林回归器结合滑动窗口策略训练预测模型，对行程量为零的网格进行需求推断。

---

## 项目结构

```
divvy/
├── data/                          # 数据目录
│   ├── chicago/                   # 芝加哥边界 Shapefile
│   ├── poi/                       # 兴趣点 Shapefile（商业、医疗、教育等10余类）
│   ├── finished_trips/            # 按周聚合的网格行程统计 CSV
│   ├── aggregated_trip_counts/    # 按车型（电动/桩式）聚合的行程统计
│   ├── dataset/                   # 预处理后的网格 POI 统计数据
│   ├── road_weight/               # 道路权重数据
│   └── material/                  # 原始站点及行程样本数据
├── src/
│   ├── process_data/              # 数据预处理脚本
│   │   ├── add_station_coord.py   # 为行程数据添加站点坐标
│   │   ├── convert_data.py        # 数据格式转换
│   │   ├── divide_by_month.py     # 按月拆分数据
│   │   ├── divide_by_week.py      # 按周拆分数据
│   │   ├── divide_by_rideable_type.py  # 按车型拆分数据
│   │   ├── divvy_json2excel.py    # 站点 JSON 转 Excel
│   │   ├── export_grid.py         # 生成地理网格
│   │   ├── tripCountPerHour.py    # 按小时聚合行程数量
│   │   ├── draw_CDF_chart.py      # 绘制 CDF 分布图
│   │   ├── draw_PDF_chart.py      # 绘制 PDF 分布图
│   │   └── ...                    # 其他辅助脚本
│   └── prediction/                # 预测模型脚本
│       ├── grid_trip.py           # 统计各网格单元的行程数量
│       ├── grid_poi.py            # 统计各网格单元的 POI 数量
│       ├── aggregate_trip_count.py # 聚合多文件行程统计
│       ├── training.py            # 模型训练（随机森林 + 滑动窗口）
│       ├── reasoning.py           # 对零行程网格进行需求推断
│       └── ...                    # 其他辅助脚本
├── output/                        # 模型输出（已在 .gitignore 中忽略）
├── model/                         # 训练好的模型文件（已在 .gitignore 中忽略）
└── readme.md
```

---

## 数据说明

| 数据目录 | 内容 |
|---|---|
| `chicago/` | 芝加哥行政边界 Shapefile，用于空间裁剪网格 |
| `poi/` | 各类城市 POI（商业、医疗、教育、体育、文化、公园、公交站、地铁站、路网、土地利用、人口属性等） |
| `finished_trips/` | 以周为单位的网格行程计数，文件命名格式：`trip_counts_week_YYYY-MM-DD.csv` |
| `aggregated_trip_counts/` | 分车型（电动/桩式）的行程统计汇总 |
| `dataset/` | 每个网格单元的 POI 分类计数，由 `grid_poi.py` 生成 |
| `road_weight/` | 道路权重/通行能力数据 |
| `material/` | 原始站点信息（JSON/XLSX）及历史行程样本（CSV/XLSM） |

---

## 技术依赖

- Python 3.x
- pandas / numpy
- geopandas / shapely
- scikit-learn（RandomForestRegressor、MinMaxScaler）
- joblib
- tqdm
- openpyxl

---

## 使用流程

### 1. 数据预处理

```bash
# 将原始行程数据按周拆分
python src/process_data/divide_by_week.py

# 为行程数据补充站点坐标
python src/process_data/add_station_coord.py
```

### 2. 空间特征生成

```bash
# 统计各 500m 网格内的行程数量（生成至 output/trip_count_week/）
python src/prediction/grid_trip.py

# 统计各 500m 网格内的 POI 数量（生成至 output/grid_poi_counts.csv）
python src/prediction/grid_poi.py

# 将 grid_poi_counts.csv 复制到 data/dataset/ 供后续使用
```

### 3. 模型训练

```bash
python src/prediction/training.py
```

训练流程：
- 以 52 周为窗口大小、逐周滑动，对历史数据执行时间序列交叉验证
- 特征：各类 POI 数量、月份、一年中第几天
- 目标变量：各网格单元的周行程数量
- 以 NMAE（归一化平均绝对误差）最低为标准保留最优模型
- 最优模型自动保存至 `model/rfr_model.pkl`

### 4. 需求推断

```bash
python src/prediction/reasoning.py
```

对训练数据中行程量为零的网格单元，使用训练好的模型预测潜在需求，结果保存至 `output/reasoning_result/`。

---

## 模型评估指标

| 指标 | 说明 |
|---|---|
| R²（决定系数） | 模型拟合优度，越接近 1 越好 |
| NMAE（归一化平均绝对误差） | MAE / 数据均值，越小越好 |
