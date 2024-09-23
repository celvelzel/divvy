import pandas as pd
import os
from tqdm import tqdm

# 指定文件夹路径
folder_path = 'E:\\作品集\\三\\数据\\新建文件夹'

# 获取文件夹中的所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 读取站点数据并将其存入字典中
stations_df = pd.read_csv('stations.csv')  # 站点数据文件
station_dict = dict(zip(stations_df['station_name'], zip(stations_df['longitude'], stations_df['latitude'])))

# 函数：处理单个骑行数据文件
def process_file(file_path, station_dict):
    # 读取骑行数据文件
    rides_df = pd.read_csv(file_path)

    # 检查是否已经包含经纬度
    if 'longitude' in rides_df.columns and 'latitude' in rides_df.columns:
        print(f"{file_path} 已包含经纬度，跳过处理。")
        return

    # 如果不包含经纬度，添加经纬度列
    rides_df['longitude'] = rides_df['station_name'].map(lambda x: station_dict.get(x, (None, None))[0])
    rides_df['latitude'] = rides_df['station_name'].map(lambda x: station_dict.get(x, (None, None))[1])

    # 保存处理后的文件
    output_file = os.path.splitext(file_path)[0] + '_with_coordinates.csv'
    rides_df.to_csv(output_file, index=False)
    print(f"{file_path} 处理完成，已保存至 {output_file}")

# 处理多个骑行数据文件
def process_multiple_files(files, station_dict):
    for file_path in tqdm(files):
        process_file(file_path, station_dict)


process_multiple_files(csv_files, station_dict)
