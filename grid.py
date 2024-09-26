import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import os
from tqdm import tqdm

# 设定你的 Excel 文件夹路径
folder_path = 'data/grid_test'

# 1. 划分网格
# 读取芝加哥边界的 Shapefile
chicago_boundary = gpd.read_file('data/chicago/芝加哥边界.shp')

# 确保边界文件是 WGS84 坐标系 (EPSG:4326)
if chicago_boundary.crs != "EPSG:4326":
    chicago_boundary = chicago_boundary.to_crs("EPSG:4326")

# 转换坐标系为适合距离计算的 UTM 投影
chicago_boundary_utm = chicago_boundary.to_crs(epsg=32616)

# 获取芝加哥的边界范围
min_x, min_y, max_x, max_y = chicago_boundary_utm.total_bounds

# 按米划分网格
grid_size = 100  # 100米

# 生成网格
polygons = []
x_coords = list(range(int(min_x), int(max_x), grid_size))
y_coords = list(range(int(min_y), int(max_y), grid_size))

for x in x_coords:
    for y in y_coords:
        polygons.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))

# 在生成网格后
print(f"Total grid polygons created: {len(polygons)}")

# 创建 GeoDataFrame 保存网格
grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:32616")

# 剔除不在芝加哥边界内的网格
grid_within_chicago = gpd.overlay(grid, chicago_boundary_utm, how='intersection')

print("generate grid finished")

# 2. 读取 Excel 数据
# 收集所有 Excel 文件的路径
excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 逐个处理 Excel 文件
for file in tqdm(excel_files):
    # 读取 Excel 文件
    poi_data = pd.read_csv(file)

    # 处理文件检查点
    print(f"processing file: {file}")

    # 将 POI 数据中的经纬度转换为地理坐标点，并检查每个 POI 属于哪个网格小方块
    poi_gdf = gpd.GeoDataFrame(poi_data,
                               geometry=gpd.points_from_xy(poi_data.longitude, poi_data.latitude),
                               crs="EPSG:4326")

    # 将 POI 数据转换为 UTM 坐标系
    poi_gdf_utm = poi_gdf.to_crs(epsg=32616)

    # 在空间连接前
    print("Starting spatial join...")
    # 使用 spatial join 将 POI 数据与网格匹配，找到每个 POI 属于哪个方块
    joined = gpd.sjoin(poi_gdf_utm, grid_within_chicago, how="left", predicate="within")
    # 在空间连接后
    print(f"Spatial join completed. Number of POIs joined: {len(joined)}")

    # 统计每个网格中的 POI 数量
    poi_counts = joined['index_right'].value_counts()  # 统计网格的索引

    # 将 POI 计数填充到网格中
    grid_within_chicago['poi_count'] = grid_within_chicago.index.map(poi_counts).fillna(0)

    # 定义输出文件名
    output_filename = f"poi_counts_{os.path.splitext(os.path.basename(file))[0]}.csv"

    # 输出每个文件的 POI 统计结果
    grid_within_chicago[['geometry', 'poi_count']].to_csv(output_filename, index=False)

    print(f"Processed and saved: {output_filename}")
