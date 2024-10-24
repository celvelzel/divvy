import joblib
import pandas as pd

# 模型保存的路径
model_path = 'model/rfc_model.pkl'
# 预测结果保存路径
output_path = 'output/reasoning_result.csv'

# 加载训练好的随机森林模型
rfc = joblib.load(model_path)
print("模型已加载。")

# 输入数据路径
data_path = 'data/dataset/grid_poi_counts.csv'
raw_data = pd.read_csv(data_path)

# 筛选 trip_count 为 0 的行
zero_trip_data = raw_data[raw_data['trip_count'] == 0]

x_new = zero_trip_data  # 仅使用行程数为0的区块数据作为输入特征
x_new.drop(columns=['grid_id', 'geometry', 'trip_count'], inplace=True)

# 进行预测
predictions = rfc.predict(x_new)

# 输出预测结果
# zero_trip_data['predicted_trip_count'] = predictions  # 将预测结果添加到新数据中
# print("预测结果：")
# print(zero_trip_data[['grid_id', 'predicted_trip_count']])  # 输出预测结果

# 保存预测结果在一个新文件中
# 将预测结果添加到原始数据中
raw_data.loc[raw_data['trip_count'] == 0, 'trip_count'] = predictions

# 保存更新后的原始数据

raw_data.to_csv(output_path, index=False)
print(f"预测结果已保存至源文件 {output_path}")
