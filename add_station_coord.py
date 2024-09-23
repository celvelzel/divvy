import pandas as pd
import os
from tqdm import tqdm

# 指定文件夹路径
folder_path = 'C:\\Users\\celcelcel\\Downloads\\divvy'
# 站点信息文件名称
stations_file = 'Divvy_Stations_2017_Q3Q4.csv'
# 站点信息文件各字段名称
station_fields = ['name', 'latitude', 'longitude']
# 行程文件站点字段名称
trip_fields = 'from_station_name'

# 创建输出文件夹
output_folder = os.path.join(folder_path, '添加站点坐标后的文件')
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f != stations_file]

# 读取站点数据并将其存入字典中
stations_df = pd.read_csv(os.path.join(folder_path, stations_file))  # 站点数据文件
station_dict = dict(zip(stations_df[station_fields[0]], zip(stations_df[station_fields[1]], stations_df[station_fields[2]])))


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
        file_path = os.path.join(folder_path, file)
        process_file(file_path, station_dict)


process_multiple_files(csv_files, station_dict)
