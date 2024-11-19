import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from statsmodels.tsa.arima.model import ARIMA
import os
import multiprocessing
from multiprocessing import Pool, Lock
import matplotlib.pyplot as plt
import warnings

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 警告屏蔽
warnings.filterwarnings("ignore")

# 设置数据文件路径
data_file = '../../output/aggregated_trip_counts/aggregated_trip_counts_docked_1.csv'
output_file = f'../../output/prediction_result/{os.path.splitext(os.path.basename(data_file))[0]}_output.csv'

# 创建文件夹
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 读取 CSV 文件
data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')

# 检查数据
# logging.info("数据预览：\n%s", data.head())

# 自定义 ARIMA 模型参数
p, d, q = 2, 0, 2
# 是否启用寻找最佳参数
enable_find_best_params = False
# 是否绘制预测结果图表
enable_plot = True
# 最佳参数列名
best_params_column_name = '1'

# 预测周数
prediction_weeks = 52

# 定义全局锁用于日志记录
lock = Lock()


def find_best_params(data_series):
    # 遍历不同的 p, d, q 组合，选择最优模型
    best_aic = float('inf')
    best_params = None

    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(data_series, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        logging.info(f"找到更好的参数组合：p={p}, d={d}, q={q}")
                except:
                    continue
    logging.info("已经找到最佳参数组合...")
    logging.info(f"最佳参数组合为：p={best_params[0]}, d={best_params[1]}, q={best_params[2]}")
    return best_params


# 定义单个时间序列的处理函数
def process_time_series(column_name):
    if len(data[column_name]) < 10:
        with lock:
            logging.warning(f"列 {column_name} 数据点少于10个，跳过。")
        return None
    try:
        # if enable_find_best_params:
        #     # 寻找最佳参数组合
        #     best_params = find_best_params(data[column_name])
        #     p = best_params[0]
        #     d = best_params[1]
        #     q = best_params[2]

        # 训练 ARIMA 模型
        logging.info(f"训练 ARIMA 模型，列 {column_name}，数据：\n{data[column_name].head()}")
        model = ARIMA(data[column_name], order=(p, d, q))
        results = model.fit()

        # 预测未来 52 步
        forecast = results.get_forecast(steps=prediction_weeks)

        # 获取预测值及其置信区间
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()

        forecast_dates = pd.date_range(start=data.index[-1], periods=prediction_weeks, freq='W')[0:] + pd.Timedelta(
            days=1)

        forecast_values.index = forecast_dates
        confidence_intervals.index = forecast_dates

        # 创建预测结果 DataFrame
        forecast_df = pd.DataFrame({column_name: forecast_values,
                                    f'Lower Bound {column_name}': confidence_intervals.iloc[:, 0],
                                    f'Upper Bound {column_name}': confidence_intervals.iloc[:, 1]},
                                   index=forecast_dates)

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

    plt.fill_between(forecast_df.index, forecast_df[f'Lower Bound {column_name}'],
                     forecast_df[f'Upper Bound {column_name}'], color='pink', alpha=0.3)

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
    if enable_find_best_params:
        # 寻找最佳参数组合
        best_params = find_best_params(data[best_params_column_name])
        p = best_params[0]
        d = best_params[1]
        q = best_params[2]

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
        if enable_plot:
            plot_forecast(original_series, forecast_df, column_name)

    # 保存最终结果
    data.to_csv(output_file, encoding='utf-8')
    logging.info(f"预测结果已成功保存到文件 {output_file}")
