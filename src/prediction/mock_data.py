import pandas as pd
import numpy as np

prediction_periods = 104

# 设置随机种子以确保结果可复现
# np.random.seed(42)

# 定义日期范围
date_range = pd.date_range(start='2020-01-01', periods=prediction_periods, freq='W')

# 定义列名
columns = ['0', '1']


# 生成按趋势增长且有周期性的数据
def generate_trend_and_seasonal_data(date_range, trend_slope, seasonal_amplitude, noise_std):
    # np.random.seed(0)  # 设置随机种子以确保结果可重现
    n_weeks = prediction_periods
    trend = np.linspace(100, 200, n_weeks)  # 模拟线性趋势
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = np.random.normal(0, 5, n_weeks)  # 添加一些噪声

    # 生成模拟数据
    return trend + seasonality + noise


# 生成数据
data = pd.DataFrame(index=date_range)

for col in columns:
    data[col] = generate_trend_and_seasonal_data(date_range, trend_slope=0.5, seasonal_amplitude=10, noise_std=5)

# 添加一些缺失值
data.iloc[::10, :] = np.nan

# 保存为 CSV 文件
data_file = '../../output/mock_data.csv'
data.to_csv(data_file, encoding='utf-8')

print("测试数据已生成并保存到文件：", data_file)
