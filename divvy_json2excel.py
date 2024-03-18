import pandas as pd
import json

# 从外部JSON文件读取数据
with open('station_20240317.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 选择需要的字段
df = df[["stationName", "bikesAvailable", "bikeDocksAvailable", "ebikesAvailable", "scootersAvailable", "totalRideablesAvailable", "isLightweight"]]

# 将数据写入Excel文件
df.to_excel("station_data_20240317.xlsx", index=False)
