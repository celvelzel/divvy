import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from statsmodels.tsa.arima.model import ARIMA
import os
import multiprocessing
from multiprocessing import Pool, Lock
import matplotlib.pyplot as plt

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置数据文件路径
data_file = '../../output/mock_data.csv'
output_file = '../../output/prediction_result/output.csv'

# 创建文件夹
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 读取 CSV 文件
data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')

# 检查数据
# logging.info("数据预览：\n%s", data.head())

# 定义 ARIMA 模型参数
p, d, q = 2, 1, 3

# 预测周数
prediction_weeks = 52

# 定义全局锁用于日志记录
lock = Lock()


# 定义单个时间序列的处理函数
def process_time_series(column_name):

    if len(data[column_name]) < 10:
        with lock:
            logging.warning(f"列 {column_name} 数据点少于10个，跳过。")
        return None

    try:
        # 训练 ARIMA 模型
        model = ARIMA(data[column_name], order=(p, d, q))
        results = model.fit()

        # 预测未来 52 步
        forecast = results.get_forecast(steps=prediction_weeks)

        # 获取预测值及其置信区间
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        # 创建预测结果 DataFrame
        forecast_df = pd.DataFrame({column_name: forecast_values,
                                    f'Lower Bound{column_name}': confidence_intervals.iloc[:, 0],
                                    f'Upper Bound{column_name}': confidence_intervals.iloc[:, 1]},
                                   index=pd.date_range(start=data.index[-1], periods=prediction_weeks+1, freq='W'))[1:]
        return forecast_df
    except Exception as e:
        with lock:
            logging.error(f"训练 ARIMA 模型时出错，列 {column_name}：{str(e)}", exc_info=True)
        return None


# 多进程处理函数
def process_data_parallel(data):
    column_names = data.columns  # 遍历列时应传递列名
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_time_series, column_names), total=len(column_names)))

    return results


# 绘制预测结果图表
def plot_forecast(original_series, forecast_df, column_name):
    plt.figure(figsize=(12, 6))

    # 绘制原始数据
    plt.plot(original_series.index, original_series, label='Original Data', color='blue')

    # 绘制预测数据
    plt.plot(forecast_df.index, forecast_df[column_name], label='Forecast', color='red', linestyle='--')

    plt.fill_between(forecast_df.index, forecast_df[f'Lower Bound{column_name}'], forecast_df[f'Upper Bound{column_name}'], color='pink', alpha=0.3)

    # 添加标题和标签
    plt.title(f'ARIMA Forecast for {column_name}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # 保存图表
    output_plot_file = f'../../output/prediction_result/plots/{column_name}_forecast.png'
    os.makedirs(os.path.dirname(output_plot_file), exist_ok=True)
    plt.savefig(output_plot_file)
    plt.close()


# 主程序
if __name__ == '__main__':
    # 获取预测结果
    logging.info("开始多进程处理时间序列数据...")
    predictions = process_data_parallel(data)

    if not predictions:
        logging.error("没有有效的预测结果，程序终止。")
        exit(1)

    # 将所有预测结果与原始数据合并
    for forecast_df in predictions:
        # 获取预测结果的列名
        column_name = forecast_df.columns[0]
        # 获取原始数据
        original_series = data[column_name].dropna()
        logging.info(f"开始处理区块 {column_name}...")

        # 将预测结果追加到原始数据末尾，并使用 reindex 进行日期对齐
        data = pd.concat([data, forecast_df], axis=0).sort_index()

        # 合并重复日期，将相同日期的第一个非空值填充
        data = data.groupby(data.index).first()

        # 绘制预测结果图表
        plot_forecast(original_series, forecast_df, column_name)

    # 保存最终结果
    data.to_csv(output_file, encoding='utf-8')
    logging.info(f"预测结果已成功保存到文件 {output_file}")
