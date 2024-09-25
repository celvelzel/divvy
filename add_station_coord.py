import pandas as pd
import os
from tqdm import tqdm

# 指定文件夹路径
trips_folder_path = 'data\\trips'
# 站点信息文件夹名称
stations_folder_path = 'data\\stations'
# 站点信息文件各字段名称
station_fields = ['name', 'latitude', 'longitude']
# 行程文件站点字段名称
trip_fields = 'from_station_name'

# 创建输出文件夹
output_folder = os.path.join(trips_folder_path, '添加站点坐标后的文件')
os.makedirs(output_folder, exist_ok=True)

# 创建年份列表：2013 到 2024
years = [str(year) for year in range(2013, 2025)]


# 函数：处理单个骑行数据文件
def process_file(file_path, station_dict):
    # 读取骑行数据文件
    rides_df = pd.read_csv(file_path)

    # 检查是否已经包含经纬度
    if 'longitude' in rides_df.columns and 'latitude' in rides_df.columns:
        print(f"{file_path} 已包含经纬度，跳过处理。")
        return

    # 如果不包含经纬度，添加经纬度列
    rides_df['latitude'] = rides_df[trip_fields].map(lambda x: station_dict.get(x, (None, None))[0])
    rides_df['longitude'] = rides_df[trip_fields].map(lambda x: station_dict.get(x, (None, None))[1])

    # 保存处理后的文件
    # 生成唯一的输出文件名
    output_file = os.path.join(output_folder, os.path.basename(file_path).replace('.csv', '_with_coordinates.csv'))
    rides_df.to_csv(output_file, index=False)
    print(f"{file_path} 处理完成，已保存至 {output_file}")


# 处理多个骑行数据文件
def process_multiple_files(files, station_dict):
    for file in tqdm(files):
        file_path = os.path.join(trips_folder_path, file)
        process_file(file_path, station_dict)


# 遍历每一年进行处理
for year in years:
    # 筛选出包含当前年份的站点数据文件
    station_files = [f for f in os.listdir(stations_folder_path) if f.endswith('.csv') and year in f]

    if not station_files:
        print(f"没有找到 {year} 的站点数据文件，跳过该年份。")
        continue

    # 读取当前年份的站点数据并将其存入字典中
    stations_df = pd.read_csv(os.path.join(stations_folder_path, station_files[0]))  # 站点数据文件
    station_dict = dict(
        zip(stations_df[station_fields[0]], zip(stations_df[station_fields[1]], stations_df[station_fields[2]])))

    # 筛选出包含当前年份的行程数据文件
    trip_files = [f for f in os.listdir(trips_folder_path) if f.endswith('.csv') and year in f]

    if not trip_files:
        print(f"没有找到 {year} 的行程数据文件，跳过该年份。")
        continue

    # 处理当前年份的行程数据文件
    process_multiple_files(trip_files, station_dict)
