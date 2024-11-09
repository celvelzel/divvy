import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from statsmodels.tsa.arima.model import ARIMA
import os
import multiprocessing
from multiprocessing import Pool, Lock

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置数据文件路径
data_file = '../../output/aggregated_trip_counts_test.csv'
output_file = '../../output/prediction_result/output.csv'
intermediate_folder = '../../output/prediction_result/intermediate'

# 创建文件夹
os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.makedirs(intermediate_folder, exist_ok=True)

# 读取 CSV 文件
data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')

# 检查数据
logging.info("数据预览：\n%s", data.head())

# 定义 ARIMA 模型参数
p, d, q = 1, 1, 1

# 定义全局锁用于日志记录
lock = Lock()


# 定义单个时间序列的处理函数
def process_time_series(column_name):
    series = data[column_name].dropna()

    if len(series) < 10:
        with lock:
            logging.warning(f"列 {column_name} 数据点少于10个，跳过。")
        return None

    try:
        # 训练 ARIMA 模型
        model = ARIMA(series, order=(p, d, q))
        results = model.fit()

        # 预测未来 52 步
        forecast = results.forecast(steps=52).tolist()

        # 生成日期索引
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=52 + 1, freq='W')[1:]  # 跳过起始日期

        # 创建预测结果 DataFrame
        forecast_df = pd.DataFrame({column_name: forecast}, index=forecast_dates)

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

    # 过滤掉 None 结果
    results = [res for res in results if res is not None]
    return results


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
        column_name = forecast_df.columns[0]

        # 将预测结果追加到原始数据末尾，并使用 reindex 进行日期对齐
        data = pd.concat([data, forecast_df], axis=0).sort_index()

        # 合并重复日期，将相同日期的第一个非空值填充
        data = data.groupby(data.index).first()

    # 保存最终结果
    data.to_csv(output_file, encoding='utf-8')
    logging.info(f"预测结果已成功保存到文件 {output_file}")
