import pandas as pd

# 加载 Excel 文件
file_path = 'C:\\Users\\celcelcel\\Desktop\\2019.8.xlsx'  # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path)

# 1. 将 B、C 两列的时间格式转换为 '年/月/日'
df['start_time'] = pd.to_datetime(df['start_time']).dt.strftime('%Y/%m/%d %H:%M:%S')
df['end_time'] = pd.to_datetime(df['end_time']).dt.strftime('%Y/%m/%d %H:%M:%S')

print("date convert finished")

# 2. 计算每一行的 C 列和 B 列之间的时间差，并将结果存储在 'm' 列
df['using time'] = (pd.to_datetime(df['end_time']) - pd.to_datetime(df['start_time'])).dt.seconds

# 3. 如果下一行的 D 列与当前行的 D 列相同，则计算下一行的 B 列和当前行的 C 列之间的时间差，并将结果存储在 'n' 列
def calculate_time_difference(row):
    if row.name < df.shape[0] - 1 and df.at[row.name + 1, 'bikeid'] == row['bikeid']:
        return (pd.to_datetime(df.at[row.name + 1, 'start_time']) - pd.to_datetime(row['end_time'])).seconds
    else:
        return None

df['non-using time'] = df.apply(calculate_time_difference, axis=1)

print("calculate finished")

# 保存更新后的数据框到新的 Excel 文件
output_file_path = 'C:\\Users\\celcelcel\\Desktop\\2019.8_s.xlsx'  # 替换为你希望保存新文件的路径
df.to_excel(output_file_path, index=False)
