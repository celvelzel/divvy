import pandas as pd

# 读取输入文件
input_file_path = '../../data/road_weight/道路权重.csv'
df = pd.read_csv(input_file_path)

# 按 grid_id 去重，保留每个 grid_id 的第一行数据
unique_df = df.drop_duplicates(subset='grid_id', keep='first')

# 重置索引，从 0 开始
unique_df.reset_index(drop=True, inplace=True)

# 提取 grid_id 和 result 列
result_df = unique_df[['grid_id', 'result']]

# 保存结果到新的 CSV 文件
output_file_path = '../../output/output_road_weight.csv'
result_df.to_csv(output_file_path, index=False)

print(f"结果已保存到 {output_file_path}")