from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
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
# 是否添加时间作为特征变量, True/False
add_time_feature = True

# 行程数据的目录
trip_file_path = '../../data/finished_trips'
# POI数据的路径
poi_file_path = '../../data/dataset/grid_poi_counts.csv'

# 配置滑动窗口参数
window_size = 52  # 窗口大小（周）
step_size = 1  # 步长（周）

# 创建输出目录
try:
    os.makedirs(model_path)
    logging.info('目录创建成功')
except FileExistsError:
    logging.info(f'目录{model_path}已经存在')


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


def load_and_merge_data(trip_data):
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

    all_trips = []

    # 读取所有行程文件
    for trip_file in trip_data:
        if trip_file.endswith('.csv'):
            csv_path = os.path.join(trip_file_path, trip_file)
            try:
                trips = pd.read_csv(csv_path)
                # 添加时间特征
                if add_time_feature:
                    trips = add_time_features(trip_file, trips)
                trips.drop(columns=['geometry'], inplace=True, errors='ignore')
                all_trips.append(trips)
            except Exception as e:
                logging.error(f"读取文件 {trip_file} 时发生错误: {e}")

    if not all_trips:
        logging.error("没有找到符合条件的文件")
        return None

    all_trips = pd.concat(all_trips, ignore_index=True)
    logging.info('行程数据拼接完成')

    # 合并行程数据和POI数据
    merged_data = pd.merge(all_trips, poi_data, on='grid_id', how='left')
    logging.info('行程数据和POI数据合并完成')

    return merged_data


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
    logging.info('数据均值: %f', y.mean())

    return x, y


def sliding_window_split(files, window_size, step_size):
    """
    使用滑动窗口按文件顺序选择训练集和测试集。

    参数：
    files (list): 按日期排序的文件路径列表
    window_size (int): 滑动窗口大小
    step_size (int): 步长

    返回：
    list: 每个窗口的训练集和测试集文件索引
    """
    file_windows = []

    for start in range(0, len(files) - window_size, step_size):
        end = start + window_size
        train_files = files[start:end]
        file_windows.append(train_files)

    return file_windows


def train_and_evaluate_model(x_train, y_train, x_test, y_test):
    """
    训练和评估模型。
    """
    # 归一化
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 训练模型
    model = RandomForestRegressor()  # 使用默认随机森林回归器
    model.fit(x_train_scaled, y_train)  # 使用训练数据集训练随机森林模型

    # 评估模型
    y_pred = model.predict(x_test_scaled)
    # importances = model.feature_importances_  # 计算特征重要性
    # importances_percent = importances * 100

    # # 打印特征重要性
    # feature_names = x_train.columns.tolist()
    # logging.info("特征重要性 (%):")
    # for feature, importance in zip(feature_names, importances_percent):
    #     logging.info(f"{feature}: {importance:.2f}%")

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # 计算数据均值
    data_mean = y_test.mean()
    # 计算NMAE
    nmae = mae / data_mean

    # logging.info(f"平均绝对误差 (MAE): {mae:.4f}")
    logging.info(f"决定系数 (R²): {r2:.4f}")
    logging.info(f"归一化平均绝对误差 (NMAE): {nmae:.4f}")

    # 保存训练好的模型
    # final_path = os.path.join(model_path, model_file_name)
    # joblib.dump(model, final_path)
    # logging.info("模型已保存")

    model_dict = {'model': model, 'r2': r2, 'nmae': nmae}
    return model_dict


def main():
    """
    主函数，执行整个流程。
    """
    # 获取按日期排序的行程文件列表
    trip_files = [f for f in os.listdir(trip_file_path) if f.endswith('.csv')]
    trip_files.sort(key=lambda f: extract_date_from_filename(f))  # 根据文件名中的日期排序

    # 使用滑动窗口划分训练集和测试集
    file_windows = sliding_window_split(trip_files, window_size, step_size)

    best_model = None
    best_r2 = 0
    best_nmae = 1

    for files_in_a_window in file_windows:
        # 根据训练集文件加载数据
        raw_data = load_and_merge_data(files_in_a_window)

        x, y = preprocess_data(raw_data)

        start_date = extract_date_from_filename(files_in_a_window[0]).date()
        end_date = extract_date_from_filename(files_in_a_window[-1]).date() + timedelta(weeks=1)
        logging.info(f"正在训练模型：窗口为{start_date}到{end_date}")
        # 训练和评估模型
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_ratio, random_state=0)
        model_dict = train_and_evaluate_model(x_train, y_train, x_test, y_test)
        model = model_dict.get('model')
        r2 = model_dict.get('r2')
        nmae = model_dict.get('nmae')

        # 保存最优模型
        if model is not None:
            # if r2 > best_r2:
            #     best_r2 = r2
            #     best_model = model
            if nmae < best_nmae:
                best_nmae = nmae
                best_r2 = r2
                best_model = model
                best_model_start_date = start_date
                best_model_end_date = end_date
                logging.info("最优模型已更新")

        # 保存最佳模型
        final_path = os.path.join(model_path, model_file_name)
    joblib.dump(best_model, final_path)
    # 打印最佳模型评估结果
    logging.info("----------------------")
    logging.info("----------------------")
    logging.info("----------------------")
    logging.info("最佳模型评估结果:")
    logging.info(f"最佳模型时间窗口为{best_model_start_date}到{best_model_end_date}")
    importances = best_model.feature_importances_  # 计算特征重要性
    importances_percent = importances * 100
    # 打印特征重要性
    feature_names = x_train.columns.tolist()
    logging.info("特征重要性 (%):")
    for feature, importance in zip(feature_names, importances_percent):
        logging.info(f"{feature}: {importance:.2f}%")
    logging.info(f"最佳模型决定系数 (R²): {best_r2:.4f}")
    logging.info(f"最佳模型归一化平均绝对误差 (NMAE): {best_nmae:.4f}")
    logging.info("最佳模型已保存")


if __name__ == "__main__":
    main()
