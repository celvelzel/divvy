import pandas as pd
import json

# 从外部JSON文件读取数据
with open('station_20240316.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 将数据转换为DataFrame
#df = pd.DataFrame(data)

# 选择需要的字段
#df = df[["stationName", "bikesAvailable", "bikeDocksAvailable", "ebikesAvailable", "scootersAvailable", "totalRideablesAvailable", "isLightweight"]]
df = pd.DataFrame([(d['stationName'], d['location']['lat'], d['location']['lng']) for d in data],
                  columns=['stationName', 'lat', 'lng'])


# 将数据写入Excel文件
df.to_excel("station_location_20240316.xlsx", index=False)
