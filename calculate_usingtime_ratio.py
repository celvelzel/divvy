import pandas as pd
import os
from tqdm.auto import tqdm

# 设置保留小数点后的位数
DECIMAL_PLACES = 20

# 定义时间阈值
THRESHOLD_24_HOURS = 24 * 3600
THRESHOLD_48_HOURS = 48 * 3600

# 指定文件夹路径
folder_path = 'C:\\Users\\celcelcel\\Desktop\\test1'  # 替换为存着所有你要处理的excel文件的的文件夹路径

# 获取文件夹中所有 Excel 文件的列表
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
tqdm.write(f"🐦‍:我开始排序 {len(files)} 个文件")

# 对每个excel文件根据bikeid排序
for file in tqdm(files):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file)

    # 加载 Excel 文件
    df = pd.read_excel(file_path)

    # 根据bikeid对整个表格排序
    df = df.sort_values(by=['bikeid', 'start_time'])

    # 保存排序后的表格
    df.to_excel(file_path, index=False)

    tqdm.write(f"🐦‍:我正在排序 {file}")

tqdm.write(f"🐦‍:我开始处理 {len(files)} 个文件")
# 遍历所有文件,使用tqdm进度条
for i, file in enumerate(tqdm(files), start=1):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file)

    # 加载 Excel 文件
    df = pd.read_excel(file_path)

    #
    # 以下是数据处理逻辑 #
    #
    # 1. 将 B、C 两列的时间格式转换为 '年/月/日'
    # 20年之前的数据格式
    df['start_time'] = pd.to_datetime(df['start_time']).dt.strftime('%Y/%m/%d %H:%M:%S')
    df['end_time'] = pd.to_datetime(df['end_time']).dt.strftime('%Y/%m/%d %H:%M:%S')

    # 20年之后的数据格式
    # df['started_at'] = pd.to_datetime(df['started_at']).dt.strftime('%Y/%m/%d %H:%M:%S')
    # df['ended_at'] = pd.to_datetime(df['ended_at']).dt.strftime('%Y/%m/%d %H:%M:%S')

    # 2. 计算每一行的 end_time 列和 start_time 列之间的时间差，并将结果存储在 using time 列
    df['using time'] = (pd.to_datetime(df['end_time']) - pd.to_datetime(df['start_time'])).dt.seconds


    # df['using time'] = (pd.to_datetime(df['ended_at']) - pd.to_datetime(df['started_at'])).dt.seconds

    # 3. 计算相邻两笔相同自行车id，到达站id等于出发站id的骑行记录之间的的时间差
    def calculate_time_difference(row):
        # 检查当前行是否为最后一行，且下一行的bikeid与当前行相同，from_station_id与当前行的to_station_id相同
        if row.name < df.shape[0] - 1 and df.at[row.name + 1, 'bikeid'] == row['bikeid'] and df.at[
            row.name + 1, "from_station_id"] == row["to_station_id"]:
            # 计算时间差并以秒为单位返回
            return (pd.to_datetime(df.at[row.name + 1, 'start_time']) - pd.to_datetime(row['end_time'])).seconds
        else:
            # 如果没有满足条件的记录，则返回None
            return None


    # def calculate_time_difference(row):
    #     # 检查当前行是否为最后一行，且下一行的bikeid与当前行相同，from_station_id与当前行的to_station_id相同
    #     if row.name < df.shape[0] - 1 and df.at[row.name + 1, 'bikeid'] == row['bikeid'] and df.at[
    #         row.name + 1, "start_station_id"] == row["end_station_id"]:
    #         # 计算时间差并以秒为单位返回
    #         return (pd.to_datetime(df.at[row.name + 1, 'start_time']) - pd.to_datetime(row['end_time'])).seconds
    #     else:
    #         # 如果没有满足条件的记录，则返回None
    #         return None

    df['non-using time'] = df.apply(calculate_time_difference, axis=1)


    # 4. 创建一个函数来检查 non-using time 是否超过特定时间
    def check_time_24(row):
        if row['non-using time'] > (THRESHOLD_24_HOURS):
            return 0
        else:
            return row['non-using time']


    def check_time_48(row):
        if row['non-using time'] > (THRESHOLD_48_HOURS):
            return 0
        else:
            return row['non-using time']


    # 5. 应用函数到每一行
    df['non-using time<24hrs'] = df.apply(check_time_24, axis=1)
    df['non-using time<48hrs'] = df.apply(check_time_48, axis=1)

    # 6. 计算 using time 和 non-using time 的总和
    total_using_time = df['using time'].sum()
    total_non_using_time = df['non-using time'].sum()
    total_non_using_time_24 = df['non-using time<24hrs'].sum()
    total_non_using_time_48 = df['non-using time<48hrs'].sum()

    # 7. 计算3种比率并保留小数点后n位
    using_time_ratio_THInfinity = round(total_using_time / (total_using_time + total_non_using_time),
                                        DECIMAL_PLACES)
    using_time_ratio_TH24 = round(total_using_time / (total_using_time + total_non_using_time_24),
                                  DECIMAL_PLACES)
    using_time_ratio_TH48 = round(total_using_time / (total_using_time + total_non_using_time_48),
                                  DECIMAL_PLACES)

    # 8. 将比率添加到新的列 "usingTime_ratio"
    df['usingTime_ratio_TH=infinity'] = pd.NA
    df['usingTime_ratio_TH=24'] = pd.NA
    df['usingTime_ratio_TH=48'] = pd.NA
    df.at[1, 'usingTime_ratio_TH=infinity'] = using_time_ratio_THInfinity
    df.at[1, 'usingTime_ratio_TH=24'] = using_time_ratio_TH24
    df.at[1, 'usingTime_ratio_TH=48'] = using_time_ratio_TH48

    # 9. 保存更新后的数据框到新的 Excel 文件
    output_file_path = os.path.join(folder_path, f"{os.path.splitext(file)[0]}_processed.xlsx")
    df.to_excel(output_file_path, index=False)

    # 进度条显示当前处理到的文件名以及预计剩余时间
    tqdm.write(f"🐦:我正在处理文件 {i}/{len(files)} : {file}")

print("All Excel files have been processed.")
