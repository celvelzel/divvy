import csv
import os

# 定义源文件夹和目标文件夹
source_folder = '../../data/trips/undivided'
target_folder_docked = '../../output/trips/docked_bike'
target_folder_electric = '../../output/trips/electric_bike'

# 确保目标文件夹存在
os.makedirs(target_folder_docked, exist_ok=True)
os.makedirs(target_folder_electric, exist_ok=True)

# 遍历源文件夹中的所有CSV文件
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(source_folder, filename)
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            docked_bikes = []
            electric_bikes = []
            for row in reader:
                if row['rideable_type'] == 'docked_bike':
                    docked_bikes.append(row)
                elif row['rideable_type'] == 'electric_bike':
                    electric_bikes.append(row)

        # 写入docked_bike数据到新的CSV文件
        docked_bikes_filename = os.path.join(target_folder_docked, f"docked_{filename}")
        with open(docked_bikes_filename, mode='w', encoding='utf-8', newline='') as docked_file:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(docked_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(docked_bikes)

        # 写入electric_bike数据到新的CSV文件
        electric_bikes_filename = os.path.join(target_folder_electric, f"electric_{filename}")
        with open(electric_bikes_filename, mode='w', encoding='utf-8', newline='') as electric_file:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(electric_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(electric_bikes)

print("CSV files have been split and saved successfully.")