import os
import pandas as pd

# 指定文件夹路径
folder_path = 'C:\\Users\\celcelcel\\Downloads\\divvy'

# 获取文件夹中的所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files.sort()  # 按文件名顺序排序，确保时间顺序正确

# 初始化变量存储上一个文件的剩余数据（即跨文件的一周）
remaining_data = pd.DataFrame()

# 创建输出文件夹
output_folder = '按周划分的文件'
os.makedirs(output_folder, exist_ok=True)

# 定义一个函数来保存按周划分的数据
def save_weekly_data(grouped_data, output_folder):
    for week, group in grouped_data:
        # 构建导出的文件名
        week_str = week.strftime('%Y-%m-%d')
        output_file = os.path.join(output_folder, f'week_{week_str}.csv')
        # 如果文件已经存在，说明这一周跨多个文件，将其追加到已存在的文件中
        if os.path.exists(output_file):
            group.to_csv(output_file, mode='a', header=False, index=False)
        else:
            group.to_csv(output_file, index=False)

# 逐个文件处理数据
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    # 读取当前文件的数据
    current_data = pd.read_csv(file_path)
    current_data['starttime'] = pd.to_datetime(current_data['starttime'], format='%Y/%m/%d %H:%M')

    # 过滤掉缺失的starttime数据
    current_data = current_data.dropna(subset=['starttime'])

    # 如果有跨文件的一周数据，将其与当前数据拼接
    if not remaining_data.empty:
        current_data = pd.concat([remaining_data, current_data])
        remaining_data = pd.DataFrame()  # 清空上一周的数据

    # 创建一个新的列表示每周的开始日期
    current_data['week_start'] = current_data['starttime'].dt.to_period('W').apply(lambda r: r.start_time)

    # 检查当前文件的最后一周是否完整
    last_week = current_data['week_start'].iloc[-1]
    last_week_data = current_data[current_data['week_start'] == last_week]
    if len(last_week_data) < 7:  # 如果最后一周数据不完整
        remaining_data = last_week_data  # 将其存储在remaining_data中
        current_data = current_data[current_data['week_start'] != last_week]  # 从当前数据中移除不完整周的数据

    # 按照每周的开始日期分组并保存数据
    grouped = current_data.groupby('week_start')
    save_weekly_data(grouped, output_folder)

# 如果最后还有剩余数据，作为最后一周保存
if not remaining_data.empty:
    remaining_grouped = remaining_data.groupby('week_start')
    save_weekly_data(remaining_grouped, output_folder)

print("按周划分的文件已成功导出到文件夹：", output_folder)
