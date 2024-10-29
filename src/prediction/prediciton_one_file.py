import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from statsmodels.tsa.arima.model import ARIMA
import os
import multiprocessing

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置数据文件路径
data_file = 'data/dateset/date.csv'  # 请修改为你的数据文件路径
output_file = 'output/prediction_result/output.csv'  # 请修改为你的输出文件路径
intermediate_folder = 'output/intermediate'  # 请修改为你的中间文件夹路径

# 创建文件夹
os.makedirs(output_file, exist_ok=True)
os.makedirs(intermediate_folder, exist_ok=True)

# 读取CSV文件
data = pd.read_csv(data_file, parse_dates=['Date'], index_col='Date')

# 检查数据
print(data.head())

# 定义ARIMA模型参数
p = 1  # 自回归项数
d = 1  # 差分阶数
q = 1  # 移动平均项数


# 定义单个时间序列的处理函数
def process_time_series(index, row):
    series = row.dropna()  # 删除缺失值
    if len(series) < 10:  # 如果数据点少于10个，跳过
        return None
    try:
        # 训练ARIMA模型
        model = ARIMA(series, order=(p, d, q))
        results = model.fit()

        # 预测未来几步
        forecast = results.forecast(steps=12)

        # 将预测结果存储到列表中
        return {
            'Index': index,
            'Forecast_1': forecast[0],
            'Forecast_2': forecast[1],
            'Forecast_3': forecast[2],
            'Forecast_4': forecast[3],
            'Forecast_5': forecast[4],
            'Forecast_6': forecast[5],
            'Forecast_7': forecast[6],
            'Forecast_8': forecast[7],
            'Forecast_9': forecast[8],
            'Forecast_10': forecast[9],
            'Forecast_11': forecast[10],
            'Forecast_12': forecast[11]
        }
    except Exception as e:
        logging.error(f"Error training ARIMA model for Index {index}: {e}")
        return None


# 多进程处理
def process_data_parallel(data):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(lambda x: process_time_series(*x), data.iterrows()), total=len(data)))

    # 过滤掉None结果
    results = [res for res in results if res is not None]
    return results


# 主程序
if __name__ == '__main__':
    # 处理数据并获取预测结果
    predictions = process_data_parallel(data)

    # 将预测结果转换为DataFrame
    predictions_df = pd.DataFrame(predictions)

    # 保存预测结果到中间文件
    intermediate_file = os.path.join(intermediate_folder, 'intermediate_results.csv')
    predictions_df.to_csv(intermediate_file, index=False)
    logging.info(f"中间结果已保存到文件 {intermediate_file}")

    # 保存预测结果到最终文件
    predictions_df.to_csv(output_file, index=False)
    logging.info(f"预测结果已成功保存到文件 {output_file}")
