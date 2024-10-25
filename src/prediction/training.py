import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, \
    mean_absolute_error, r2_score
import joblib
import os

# 数据路径
# data_path = 'data/dataset/grid_poi_counts.csv'
# 模型保存路径
model_path = '../../model/rfc_model.pkl'
# 测试集比例
text_size_ratio = 0.1
# 额外的0行程网格比例
zero_trip_ratio = 0.1

# 多周行程与poi根据区块索引进行拼接
# 行程数据的目录
trip_file_path = '../../data/trips'
# POI数据的路径
poi_file_path = '../../output/grid_poi_counts.csv'

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
        trips.drop(columns=['geometry'], inplace=True)
        all_trips.append(trips)
all_trips = pd.concat(all_trips, ignore_index=True)

# 合并行程数据和POI数据
merged_data = pd.merge(all_trips, poi_data, on='grid_id', how='left')
raw_data = merged_data
print('行程数据拼接完成')
print(merged_data.head())

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

# 删除不需要的列
# train_data.drop(columns=['grid_id', 'geometry'], inplace=True)
train_data.drop(columns=['grid_id'], inplace=True)

# 提取输入特征和目标变量
x = train_data.drop('trip_count', axis=1)  # 输入特征
y = train_data['trip_count']  # 目标变量

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=text_size_ratio, random_state=0)

# 训练模型
rfc = RandomForestClassifier()  # 使用默认随机森林分类器
rfc.fit(x_train, y_train)  # 使用训练数据集训练随机森林模型

# 评估模型
y_pred = rfc.predict(x_test)  # 使用分类器预测测试集的类别
importances = rfc.feature_importances_  # 计算特征重要性
print("Importances")
print(importances)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))  # 输出混淆矩阵
print("Classification Report:")
print(classification_report(y_test, y_pred))  # 输出分类报告
mse = mean_squared_error(y_test, y_pred)
# print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {np.sqrt(mse):.4f}")
print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
# print(f"决定系数 (R²): {r2_score(y_test, y_pred):.4f}")
print("Accuracy:")  # 输出精度
print(accuracy_score(y_test, y_pred))

# 保存训练好的模型
joblib.dump(rfc, model_path)
print("模型已保存")
