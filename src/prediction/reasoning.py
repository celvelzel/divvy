import joblib
import pandas as pd
from tqdm import tqdm
import re
import os

# 模型保存的路径
model_path = '../../model/rfc_model.pkl'
# 预测结果保存路径
output_path = '../../output/reasoning_result'

# 输入数据路径
poi_data_path = '../../output/grid_poi_counts.csv'  # POI 数据路径
trip_data_path = '../../data/dataset/trip_count'  # 行程数据路径

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

for trip_file_path in tqdm(trip_data_path):
    if trip_data_path.endswith('.csv'):
        data_path = os.path.join(trip_data_path, trip_file_path)

        # 加载行程数据
        trip_data = pd.read_csv(data_path)
        if 'geometry' in trip_data.columns:
            trip_data.drop(columns=['geometry'], inplace=True)

        # 合并行程数据和POI数据
        raw_data = pd.merge(poi_data, trip_data, on='grid_id', how='right')

        # 筛选 trip_count 为 0 的行
        zero_trip_data = raw_data[raw_data['trip_count'] == 0]

        x_new = zero_trip_data  # 仅使用行程数为0的区块数据作为输入特征
        x_new.drop(columns=['grid_id', 'trip_count'], inplace=True)

        # 进行预测
        predictions = rfc.predict(x_new)

        # 输出预测结果
        # zero_trip_data['predicted_trip_count'] = predictions  # 将预测结果添加到新数据中
        # print("预测结果：")
        # print(zero_trip_data[['grid_id', 'predicted_trip_count']])  # 输出预测结果

        # 将预测结果添加到原始数据中
        raw_data.loc[raw_data['trip_count'] == 0, 'trip_count'] = predictions

        # 定义输出文件名
        filename = os.path.splitext(os.path.basename(data_path))[0]
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        date = re.search(date_pattern, filename)
        output_path= f"../../output/reasoning_result/reasoning_result_{date}.csv"

        # 保存更新后的原始数据
        raw_data.to_csv(output_path, index=False)
        print(f"预测结果已保存至文件 {output_path}")

