import geopandas as gpd
import pandas as pd

# # 读取 POI 数据
# poi_data = pd.read_csv("data/2013.9_with_coordinates.csv")  # 假设 POI 数据在 CSV 中
#
# # 创建 GeoDataFrame，确保使用 EPSG:4326（WGS84 经纬度坐标系）
# poi_gdf = gpd.GeoDataFrame(poi_data,
#                            geometry=gpd.points_from_xy(poi_data.longitude, poi_data.latitude),
#                            crs="EPSG:4326")

# 检查 POI 数据的 CRS
# EPSG:4326
# print(poi_gdf.crs)

# 读取 Shapefile 文件
shapefile_gdf = gpd.read_file("data/芝加哥边界.shp")

# 检查 Shapefile 的 CRS
print(shapefile_gdf.crs)
