import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, \
    mean_absolute_error, r2_score
import joblib
import os
import re
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# 数据路径
# data_path = 'data/dataset/grid_poi_counts.csv'
# 模型保存路径
model_path = '../../model'
# 模型文件名
model_file_name = 'rfc_model.pkl'
# 测试集比例
test_size_ratio = 0.3
# 额外的0行程网格比例
zero_trip_ratio = 0.01

# 多周行程与poi根据区块索引进行拼接
# 行程数据的目录
trip_file_path = '../../data/trips'
# POI数据的路径
poi_file_path = '../../data/dataset/grid_poi_counts.csv'

# 创建输出目录
try:
    os.makedirs(model_path)
    print('目录创建成功')
except FileExistsError:
    print(f'目录{model_path}已经存在')


def add_time(trip_file, trips):
    # 从文件名提取日期
    date_str = trip_file.split('_')[-1].replace('.csv', '')  # 获取日期字符串
    matcher = re.compile(r'\d{4}-\d{2}-\d{2}')
    date_str = matcher.search(date_str).group()
    date = pd.to_datetime(date_str)  # 转换为日期格式

    # 提取特征
    # trips['date'] = date
    trips['month'] = date.month  # 月份
    trips['day_of_year'] = date.dayofyear  # 一年中的第几天

    # 可以考虑将月份和周几转换为周期性特征
    # trips['sin_month'] = np.sin(2 * np.pi * date.month / 12)
    # trips['sin_day_of_week'] = np.sin(2 * np.pi * date.dayofyear / 7)

    return trips


# 读取poi数据
poi_data = pd.read_csv(poi_file_path)
poi_data.drop(columns=['geometry'], inplace=True)
# 将每周行程文件拼接
all_trips = []
# 遍历文件夹中所有文件
for trip_file in os.listdir(trip_file_path):
    if trip_file.endswith('.csv'):
        csv_path = os.path.join(trip_file_path, trip_file)
        trips = pd.read_csv(csv_path)
        trips = add_time(trip_file, trips)
        trips.drop(columns=['geometry'], inplace=True)
        all_trips.append(trips)
all_trips = pd.concat(all_trips, ignore_index=True)

# 合并行程数据和POI数据
merged_data = pd.merge(all_trips, poi_data, on='grid_id', how='left')
print('行程数据拼接完成')

# 计算每个区块的平均行程数
avg_trip_count = merged_data.groupby('grid_id')['trip_count'].mean().reset_index()
avg_trip_count.columns = ['grid_id', 'avg_trip_count']

# 合并平均行程数到原始数据
merged_data = pd.merge(merged_data, avg_trip_count, on='grid_id', how='left')
print('平均行程数据拼接完成')
raw_data = merged_data
print(raw_data.head())
# raw_data.to_csv('../../output/raw_data.csv', index=False)

# 加载单个周行程文件
# raw_data = pd.read_csv(path)  # 加载数据

# 筛选行程不为0的数据
non_zero_trip_grid = raw_data[raw_data['trip_count'] > 0]

# 筛选行程为0的数据
zero_trip_grid = raw_data[raw_data['trip_count'] == 0]

# 随机抽取一定比例的0行程网格数据
num_zero_samples = int(len(non_zero_trip_grid) * zero_trip_ratio)
sampled_zero_trips = zero_trip_grid.sample(n=num_zero_samples, random_state=0)

# 合并数据
combined_data = pd.concat([non_zero_trip_grid, sampled_zero_trips])

train_data = combined_data.copy()

# 输出数据平均值
print('数据集trip_count平均值:')
print(train_data['trip_count'].mean())

# 删除不需要的列
# train_data.drop(columns=['grid_id', 'geometry'], inplace=True)
train_data.drop(columns=['grid_id'], inplace=True)

# 提取输入特征和目标变量
x = train_data.drop(['trip_count', 'avg_trip_count'], axis=1)  # 输入特征
y = train_data['trip_count']  # 目标变量

# 列名
feature_names = x.columns.tolist()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_ratio, random_state=0)

# 归一化
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# try:
#     # 网格参数
#     param_grid = {
#         'n_estimators': [100, 200, 300, 400, 500],
#         'max_depth': [None, 10, 20, 30, 40],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4, 6]
#     }
#     grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3)
#     grid_search.fit(x_train_scaled, y_train)
#     print("最佳参数：", grid_search.best_params_)
# except Exception as e:
#     print(e)

# 训练模型
# 最佳参数： {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
# params = grid_search.best_params_
# model = RandomForestRegressor(**params)  # 使用默认随机森林回归器
model = RandomForestRegressor()  # 使用默认随机森林回归器
model.fit(x_train_scaled, y_train)  # 使用训练数据集训练随机森林模型

# 评估模型
y_pred = model.predict(x_test_scaled)

importances = model.feature_importances_  # 计算特征重要性
# 将特征重要性转换为百分比形式
importances_percent = importances * 100
# 打印特征重要性
print("Feature Importances (%):")
for feature, importance in zip(feature_names, importances_percent):
    print(f"{feature}: {importance:.2f}%")

mse = mean_squared_error(y_test, y_pred)
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {np.sqrt(mse):.4f}")
print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
print(f"决定系数 (R²): {r2_score(y_test, y_pred):.4f}")
# print("Accuracy:")  # 输出精度
# print(accuracy_score(y_test, y_pred))

# 保存训练好的模型
final_path = os.path.join(model_path, model_file_name)
joblib.dump(model, final_path)
print("模型已保存")
