import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from statsmodels.tsa.arima.model import ARIMA
import os
import multiprocessing
from multiprocessing import Pool

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置数据文件路径
data_file = '../../output/aggregated_trip_counts.csv'
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


# 定义单个时间序列的处理函数
def process_time_series(index, row):
    series = row.dropna()
    if len(series) < 10:
        logging.warning(f"Index {index} 数据点少于10个，跳过。")
        return None
    try:
        # 训练 ARIMA 模型
        model = ARIMA(series, order=(p, d, q))
        results = model.fit()

        # 预测未来 52 步
        forecast = results.forecast(steps=52).tolist()

        # 构建预测结果字典
        forecast_dict = {'Index': index}
        forecast_dict.update({f'Forecast_{i + 1}': val for i, val in enumerate(forecast)})

        return forecast_dict
    except Exception as e:
        logging.error(f"训练 ARIMA 模型时出错，Index {index}：{str(e)}", exc_info=True)
        return None


# 定义用于多进程的辅助函数
def process_row(args):
    return process_time_series(*args)


# 多进程处理函数
def process_data_parallel(data):
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_row, data.iterrows()), total=len(data)))

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

    # 将预测结果转换为 DataFrame
    predictions_df = pd.DataFrame(predictions)

    # 保存中间结果
    intermediate_file = os.path.join(intermediate_folder, 'intermediate_results.csv')
    predictions_df.to_csv(intermediate_file, index=False, encoding='utf-8')
    logging.info(f"中间结果已保存到文件 {intermediate_file}")

    # 保存最终结果
    predictions_df.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"预测结果已成功保存到文件 {output_file}")
