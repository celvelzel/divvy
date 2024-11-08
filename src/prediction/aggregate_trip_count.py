import os
import pandas as pd
import re
from tqdm import tqdm

date_pattern = r'\d{4}-\d{2}-\d{2}'

# 定义输入文件夹路径和输出文件路径
input_folder = '../../output/reasoning_result'
output_file = '../../output/aggregated_trip_counts.csv'

# 初始化空列表，用于存储每个周的数据
weekly_data = []

# 遍历文件夹中的每个 CSV 文件
for file_name in tqdm(sorted(os.listdir(input_folder))):
    if file_name.endswith('.csv'):
        # 提取周标识，例如 'week1' 等，作为行索引
        match = re.search(date_pattern, file_name)
        if match:
            date = match.group()
            week = date

        # 读取 CSV 文件
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path, usecols=['grid_id', 'trip_count'])

        # 将区块数据转换为一行，以 `grid_id` 作为列名，`trip_counts` 作为值
        week_series = df.set_index('grid_id')['trip_count']
        week_series.name = week  # 行名设置为周标识

        # 将周数据追加到列表中
        weekly_data.append(week_series)

# 将所有周数据合并为一个 DataFrame
summary_df = pd.DataFrame(weekly_data)

# 缺失值填充为 0（如果某些周的某个区块没有数据）
summary_df = summary_df.fillna(0)

# 保存汇总后的数据到 CSV 文件
summary_df.to_csv(output_file)
print(f"数据已成功汇总到文件 {output_file}")
