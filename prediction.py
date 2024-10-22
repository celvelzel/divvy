import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 设置行程数据的起始日期
start_date = '2013-01-01'
# 设置行程数据的结束时间
end_date = '2013-12-31'
# 闭区间
dates = pd.date_range(start=start_date, end_dates=end_date, freq='7D')

# 设置数据文件夹路径
data_folder = 'data/dateset'  # 请修改为你的数据文件夹路径

# 获取所有CSV文件
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# 检查是否有CSV文件
if not csv_files:
    raise ValueError("未找到CSV文件，请检查数据文件夹路径。")

# 初始化一个列表来存储所有周的行程总量
total_trips_list = []

# 遍历所有文件，读取数据并合并
for csv_file in tqdm(csv_files):
    file_path = os.path.join(data_folder, csv_file)

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查数据格式
        if df.shape[1] < 2:
            raise ValueError(f"{csv_file} 格式不正确，至少需要两列。")

        # 假设第一列为索引，第二列为行程数
        trips = df.iloc[:, 1]

        # 2. 数据预处理
        try:
            # 检查是否有缺失值
            if data.isnull().sum().sum() > 0:
                logging.warning("Interpolating missing values.")
                data = data.interpolate()

            # 可视化数据
            # plt.figure(figsize=(12, 6))
            # plt.plot(data.index, data['TotalTrips'])
            # plt.xlabel('Date')
            # plt.ylabel('Total Trips')
            # plt.title('Weekly Total Trips Over 9 Years')
            # plt.show()
        except Exception as e:
            logging.error(f"Error in data preprocessing or visualization: {e}")
            raise

    except Exception as e:
        print(f"处理文件 {csv_file} 时发生错误: {e}")

# 创建时间序列
if not total_trips_list:
    raise ValueError("未能生成行程总量，请检查输入数据。")

weekly_totals = pd.Series(total_trips_list)
weekly_totals.index = pd.date_range(start=start_date, periods=len(weekly_totals), freq='7D')  # 请根据实际情况修改起始日期

# 可视化数据
plt.figure(figsize=(12, 6))
plt.plot(weekly_totals, marker='o', linestyle='-')
plt.title('每周共享单车行程总量')
plt.xlabel('日期')
plt.ylabel('行程数')
plt.grid()
plt.show()

# ACF和PACF图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(weekly_totals, ax=ax[0])
plot_pacf(weekly_totals, ax=ax[1])
plt.show()

# 建立ARIMA模型
# 这里建议使用p, d, q的参数可以根据ACF/PACF图手动选择
p = 1  # 根据PACF图选择
d = 1  # 根据数据平稳性选择
q = 1  # 根据ACF图选择

try:
    model = ARIMA(weekly_totals, order=(p, d, q))
    model_fit = model.fit()

    # 进行未来几周的预测
    forecast_steps = 4
    forecast = model_fit.forecast(steps=forecast_steps)

    # 输出预测结果
    print("未来几周的预测行程数:")
    print(forecast)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_totals, label='实际行程数', marker='o')
    plt.plot(forecast.index, forecast, label='预测行程数', color='red', marker='o')
    plt.title('共享单车行程预测')
    plt.xlabel('日期')
    plt.ylabel('行程数')
    plt.legend()
    plt.grid()
    plt.show()

    # 残差分析
    residuals = model_fit.resid
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title('残差分析')
    plt.xlabel('日期')
    plt.ylabel('残差值')
    plt.grid()
    plt.show()

    # 进行Ljung-Box检验
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print("Ljung-Box检验结果:")
    print(lb_test)

except Exception as e:
    print(f"模型建立和预测时发生错误: {e}")
