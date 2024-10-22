import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. 随机生成572个有趋势的数据
np.random.seed(42)  # 设置随机种子以确保结果可重现
n_weeks = 572  # 9年每周的总数
trend = np.linspace(100, 200, n_weeks)  # 模拟线性趋势
seasonality = 20 * np.sin(np.linspace(0, 3 * np.pi, n_weeks))  # 模拟季节性
noise = np.random.normal(0, 5, n_weeks)  # 添加一些噪声

# 生成模拟数据
data = trend + seasonality + noise

# 将数据转换为 DataFrame
dates = pd.date_range(start='2015-01-01', periods=n_weeks, freq='W')
df = pd.DataFrame(data, index=dates, columns=['Trips'])

# 2. 数据可视化
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Trips'], label='历史行程数', color='blue')
plt.title('模拟的单车行程总量')
plt.xlabel('日期')
plt.ylabel('行程数')
plt.legend()
plt.grid()
plt.show()

# 3. 构建并训练 ARIMA 模型
model = ARIMA(df['Trips'], order=(1, 1, 1))  # 选择合适的 p, d, q
model_fit = model.fit()

# 4. 进行预测未来52周
forecast_steps = 52
forecast = model_fit.forecast(steps=forecast_steps)

# 生成未来日期
last_date = df.index[-1]  # 获取最后一个日期
future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W')

# 创建预测结果的 DataFrame
forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicted_Trips'])

# 5. 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Trips'], label='历史行程数', color='blue')
plt.plot(forecast_df.index, forecast_df['Predicted_Trips'], label='预测行程数', color='red', marker='o')
plt.title('单车行程数量预测')
plt.xlabel('日期')
plt.ylabel('行程数')
plt.legend()
plt.grid()
plt.show()

# 输出预测结果
print(f"未来52周的预测结果：\n{forecast_df}")
