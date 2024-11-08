import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re
import os
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 模型保存的路径
model_path = '../../model/rfc_model.pkl'
# 预测结果保存路径
output_path = '../../output/reasoning_result'
# 添加时间特征
add_time_feature = True

# 输入数据路径
poi_data_path = '../../data/dataset/grid_poi_counts.csv'  # POI 数据路径
trip_data_path = '../../data/finished_trips'  # 行程数据路径

# 加载 POI 数据
poi_data = pd.read_csv(poi_data_path)
if 'geometry' in poi_data.columns:
    poi_data.drop(columns=['geometry'], inplace=True)

# 加载训练好的随机森林模型
rfc = joblib.load(model_path)
print("模型已加载")

# 创建输出目录
try:
    os.makedirs(output_path)
    print('目录创建成功')
except FileExistsError:
    print(f'目录{output_path}已经存在')


def extract_date_from_filename(filename):
    """
    从文件名中提取日期。
    假设文件名格式为 'trips_YYYY-MM-DD.csv'
    """
    matcher = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if matcher:
        return pd.to_datetime(matcher.group())
    return None


def add_time_features(trip_file, trips):
    """
    从文件名中提取日期，并添加时间特征。

    参数:
    trip_file (str): 文件名
    finished_trips (DataFrame): 行程数据

    返回:
    DataFrame: 添加了时间特征的行程数据
    """
    try:
        date = extract_date_from_filename(trip_file)
        trips['month'] = date.month  # 月份
        trips['day_of_year'] = date.dayofyear  # 一年中的第几天
        return trips
    except Exception as e:
        logging.error(f"处理文件 {trip_file} 时发生错误: {e}")
        return trips


trip_files = [f for f in os.listdir(trip_data_path) if f.endswith('.csv')]
for trip_file in tqdm(trip_files):
    print(trip_file)
    if trip_file.endswith('.csv'):
        print(f"正在处理文件：{trip_file}")
        data_path = os.path.join(trip_data_path, trip_file)

        # 加载行程数据
        trip_data = pd.read_csv(data_path)
        # 添加时间特征
        if add_time_feature:
            trip_data = add_time_features(trip_file, trip_data)
        if 'geometry' in trip_data.columns:
            trip_data.drop(columns=['geometry'], inplace=True)

        # 合并行程数据和POI数据
        raw_data = pd.merge(poi_data, trip_data, on='grid_id', how='right')

        # 筛选 trip_count 为 0 的行
        zero_trip_data = raw_data[raw_data['trip_count'] == 0]

        x_new = zero_trip_data  # 仅使用行程数为0的区块数据作为输入特征
        x_new.drop(columns=['grid_id', 'trip_count'], inplace=True)

        # 归一化
        scaler = MinMaxScaler()
        x_new_scaled = scaler.fit_transform(x_new)

        # 进行预测
        predictions = rfc.predict(x_new_scaled)

        # 输出预测结果
        # zero_trip_data['predicted_trip_count'] = predictions  # 将预测结果添加到新数据中
        # print("预测结果：")
        # print(zero_trip_data[['grid_id', 'predicted_trip_count']])  # 输出预测结果

        # 将预测结果添加到原始数据中
        raw_data.loc[raw_data['trip_count'] == 0, 'trip_count'] = predictions

        # 定义输出文件名
        filename = os.path.splitext(os.path.basename(data_path))[0]
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        match = re.search(date_pattern, filename)
        if match:
            date = match.group()
            output_path = f"../../output/reasoning_result/reasoning_result_{date}.csv"

        # 保存更新后的原始数据
        raw_data.to_csv(output_path, index=False)
        print(f"预测结果已保存至文件 {output_path}")
    else:
        print('csv file not found')
