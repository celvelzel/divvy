import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import re
from sklearn.preprocessing import MinMaxScaler
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据路径
# data_path = 'data/dataset/grid_poi_counts.csv'
# 模型保存路径
model_path = '../../model'
# 模型文件名
model_file_name = 'rfc_model.pkl'
# 测试集比例
test_size_ratio = 0.3
# 额外的0行程网格比例
zero_trip_ratio = 0
# 是否添加时间作为特征变量
add_time_feature = True
# 训练集和测试集的划分日期
divide_date = '2020-12-28'

# 行程数据的目录
trip_file_path = '../../data/finished_trips/training'
# 测试数据的目录
test_file_path = '../../data/finished_trips/test'
# POI数据的路径
poi_file_path = '../../data/dataset/grid_poi_counts.csv'

# 创建输出目录
try:
    os.makedirs(model_path)
    logging.info('目录创建成功')
except FileExistsError:
    logging.info(f'目录{model_path}已经存在')


def add_time_features(trip_file, trips):
    """
    从文件名中提取日期，并添加时间特征。

    参数:
    trip_file (str): 文件名
    trips (DataFrame): 行程数据

    返回:
    DataFrame: 添加了时间特征的行程数据
    """
    try:
        date_str = trip_file.split('_')[-1].replace('.csv', '')
        matcher = re.compile(r'\d{4}-\d{2}-\d{2}')
        date_str = matcher.search(date_str).group()
        date = pd.to_datetime(date_str)

        trips['month'] = date.month  # 月份
        trips['day_of_year'] = date.dayofyear  # 一年中的第几天
        return trips
    except Exception as e:
        logging.error(f"处理文件 {trip_file} 时发生错误: {e}")
        return trips


def is_exceed_date(trip_file):
    """
    参数:
    trip_file (str): 文件名

    判断该行程文件是否超过划分日期，
    如果超过为测试集，如果未超过为训练集
    """
    try:
        date_str = trip_file.split('_')[-1].replace('.csv', '')
        matcher = re.compile(r'\d{4}-\d{2}-\d{2}')
        date_str = matcher.search(date_str).group()
        date = pd.to_datetime(date_str)

        if date > pd.to_datetime(divide_date):
            return True
    except Exception as e:
        logging.error(f"处理文件 {trip_file} 时发生错误: {e}")
        return False


def load_and_process_data():
    """
    加载和处理数据。

    返回:
    DataFrame: 处理后的数据
    """
    # 读取poi数据
    try:
        poi_data = pd.read_csv(poi_file_path)
        poi_data.drop(columns=['geometry'], inplace=True, errors='ignore')
    except Exception as e:
        logging.error(f"读取POI数据时发生错误: {e}")
        return None

    # 将每周行程文件拼接
    training_set = []
    testing_set = []
    trips_file_dict = []
    for trip_file in os.listdir(trip_file_path):
        if trip_file.endswith('.csv'):
            csv_path = os.path.join(trip_file_path, trip_file)
            # 未超过分割日期的为训练集
            if not is_exceed_date(trip_file):
                try:

                    trips_file_dict.append({'file_path': csv_path, 'category': 'train'})
                except Exception as e:
                    logging.error(f"读取文件 {trip_file} 时发生错误: {e}")
            else:
                trips_file_dict.append({'file_path': csv_path, 'category': 'test'})

    # 遍历字典
    for item in trips_file_dict:
        csv_path = item['file_path']
        trips = pd.read_csv(csv_path)
        # 添加时间特征
        if add_time_feature:
            trips = add_time_features(csv_path, trips)
        trips.drop(columns=['geometry'], inplace=True, errors='ignore')
        if item['category'] == 'test':
            testing_set.append(trips)
        else:
            training_set.append(trips)

    if not training_set:
        logging.error("没有找到符合条件的文件")
        return None

    # 拼接所有数据
    training_set = pd.concat(training_set, ignore_index=True)
    testing_set = pd.concat(testing_set, ignore_index=True)
    logging.info('行程数据拼接完成')

    # 合并行程数据和POI数据
    dataset = []
    merged_training_data = pd.merge(training_set, poi_data, on='grid_id', how='left')
    dataset.append({'category': 'train', 'data': merged_training_data})
    merged_testing_data = pd.merge(testing_set, poi_data, on='grid_id', how='left')
    dataset.append({'category': 'test', 'data': merged_testing_data})
    logging.info('行程数据和POI数据合并完成')

    # 计算每个区块的平均行程数
    # avg_trip_count = merged_data.groupby('grid_id')['trip_count'].mean().reset_index()
    # avg_trip_count.columns = ['grid_id', 'avg_trip_count']
    # merged_data = pd.merge(merged_data, avg_trip_count, on='grid_id', how='left')
    # logging.info('平均行程数据拼接完成')

    return dataset


def preprocess_data(data):
    """
    预处理数据。

    参数:
    data (DataFrame): 原始数据

    返回:
    DataFrame: 预处理后的数据
    """
    # 筛选行程不为0的数据
    non_zero_trip_grid = data[data['trip_count'] > 0]

    # 筛选行程为0的数据
    zero_trip_grid = data[data['trip_count'] == 0]

    # 随机抽取一定比例的0行程网格数据
    num_zero_samples = int(len(non_zero_trip_grid) * zero_trip_ratio)
    sampled_zero_trips = zero_trip_grid.sample(n=num_zero_samples, random_state=0)

    # 合并数据
    combined_data = pd.concat([non_zero_trip_grid, sampled_zero_trips])

    # 删除不需要的列
    combined_data.drop(columns=['grid_id'], inplace=True, errors='ignore')

    # 提取输入特征和目标变量
    # x = combined_data.drop(['trip_count', 'avg_trip_count'], axis=1)  # 输入特征
    x = combined_data.drop(['trip_count'], axis=1)  # 输入特征
    y = combined_data['trip_count']  # 目标变量

    # 输出数据平均值
    logging.info('数据集trip_count平均值:')
    logging.info(y.mean())

    return x, y


def train_and_evaluate_model(x, y):
    """
    训练和评估模型。

    参数:
    x (DataFrame): 输入特征
    y (Series): 目标变量
    """
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_ratio, random_state=0)

    # 归一化
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 训练模型
    model = RandomForestRegressor()  # 使用默认随机森林回归器
    model.fit(x_train_scaled, y_train)  # 使用训练数据集训练随机森林模型

    # 评估模型
    y_pred = model.predict(x_test_scaled)
    importances = model.feature_importances_  # 计算特征重要性
    importances_percent = importances * 100

    # 打印特征重要性
    feature_names = x.columns.tolist()
    logging.info("特征重要性 (%):")
    for feature, importance in zip(feature_names, importances_percent):
        logging.info(f"{feature}: {importance:.2f}%")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"均方误差 (MSE): {mse:.4f}")
    logging.info(f"均方根误差 (RMSE): {rmse:.4f}")
    logging.info(f"平均绝对误差 (MAE): {mae:.4f}")
    logging.info(f"决定系数 (R?): {r2:.4f}")

    # 保存训练好的模型
    final_path = os.path.join(model_path, model_file_name)
    joblib.dump(model, final_path)
    logging.info("模型已保存")


def main():
    """
    主函数，执行整个流程。
    """
    # 加载和处理数据
    raw_data = load_and_process_data()
    if raw_data is None:
        logging.error("数据加载失败")
        return

    # 预处理数据
    x, y = preprocess_data(raw_data)

    # 训练和评估模型
    train_and_evaluate_model(x, y)

    print("训练结束")


if __name__ == "__main__":
    main()
