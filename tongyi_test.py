import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging
from datetime import datetime

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start = datetime.now()

# 1. 生成模拟数据
try:
    # np.random.seed(0)
    dates = pd.date_range(start='2013-01-01', periods=572, freq='W')
    # base_trend = np.linspace(1000, 3000, 572)  # 生成一个线性趋势
    # noise = np.random.normal(0, 100, 572)  # 添加一些随机噪声
    # total_trips = base_trend + noise

    np.random.seed(42)  # 设置随机种子以确保结果可重现
    n_weeks = 572  # 9年每周的总数
    trend = np.linspace(100, 200, n_weeks)  # 模拟线性趋势
    # seasonality = 20 * np.sin(np.linspace(0, 3 * np.pi, n_weeks))  # 模拟季节性
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = np.random.normal(0, 5, n_weeks)  # 添加一些噪声

    # 生成模拟数据
    total_trips = trend + seasonality + noise

    # 创建DataFrame
    data = pd.DataFrame({'Date': dates, 'TotalTrips': total_trips})
    data.set_index('Date', inplace=True)

    # 查看数据
    logging.info("Data generation complete.")
    print(data.head())
except Exception as e:
    logging.error(f"Error generating data: {e}")
    raise

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

# 3. 训练ARIMA模型
try:
    # 定义ARIMA模型参数
    p, d, q = 2, 1, 3

    # 训练ARIMA模型
    model = ARIMA(data['TotalTrips'], order=(p, d, q))
    results = model.fit()

    # 打印模型摘要
    # logging.info("Model training complete.")
    # print(results.summary())
except Exception as e:
    logging.error(f"Error training ARIMA model: {e}")
    raise

# 4. 预测未来52周的数据
try:
    # 预测未来52周的行程总量
    forecast_steps = 52
    forecast = results.get_forecast(steps=forecast_steps)

    # 获取预测值及其置信区间
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # 将预测值和置信区间添加到DataFrame中
    forecast_df = pd.DataFrame({'Forecast': forecast_values,
                                'Lower Bound': confidence_intervals.iloc[:, 0],
                                'Upper Bound': confidence_intervals.iloc[:, 1]},
                               index=pd.date_range(start=data.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W'))

    # 查看预测结果
    logging.info("Forecasting complete.")
    print(forecast_df.head())
except Exception as e:
    logging.error(f"Error in forecasting: {e}")
    raise

finished = datetime.now()
print("Total time taken: ", finished - start)

# 5. 绘制预测结果
try:
    # 绘制原始数据和预测结果
    plt.figure(figsize=(12, 6))

    # 绘制原始数据
    plt.plot(data.index, data['TotalTrips'], label='Actual Total Trips')

    # 绘制预测结果
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecasted Total Trips', linestyle='--', color='red')
    plt.fill_between(forecast_df.index, forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='pink', alpha=0.3)

    plt.xlabel('Date')
    plt.ylabel('Total Trips')
    plt.title('Weekly Total Trips Forecast')
    plt.legend()
    plt.show()
except Exception as e:
    logging.error(f"Error in plotting forecast results: {e}")
    raise


# 6. 评估模型性能
# try:
#     # 假设我们有一个测试集来评估模型性能
#     test_data = pd.read_csv('test_data.csv', parse_dates=['Date'], index_col='Date')
#
#     # 计算预测值和真实值之间的均方根误差（RMSE）
#     predictions = results.forecast(steps=len(test_data))
#     rmse = sqrt(mean_squared_error(test_data['TotalTrips'], predictions))
#     logging.info(f"Root Mean Squared Error: {rmse}")
# except Exception as e:
#     logging.error(f"Error in model evaluation: {e}")
#     raise