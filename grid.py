import geopandas as gpd
from shapely.geometry import Polygon, Point
import pandas as pd
import os
from tqdm import tqdm

# 设定你的 Excel 文件夹路径
folder_path = 'path/to/excel/files'

# 1.划分网格
# 读取芝加哥边界的 Shapefile
chicago_boundary = gpd.read_file('path/to/chicago_boundary.shp')

# 确保边界文件是 WGS84 坐标系 (EPSG:4326)
if chicago_boundary.crs != "EPSG:4326":
    chicago_boundary = chicago_boundary.to_crs("EPSG:4326")

# 转换坐标系为适合距离计算的 UTM 投影
# UTM 投影用于精确的米制单位，芝加哥处于 UTM 16N (EPSG:32616)
chicago_boundary_utm = chicago_boundary.to_crs(epsg=32616)

# 获取芝加哥的边界范围
min_x, min_y, max_x, max_y = chicago_boundary_utm.total_bounds

# 3. 按米划分网格
grid_size = 1000  # 100米

# 生成网格
x_coords = list(range(int(min_x), int(max_x), grid_size))
y_coords = list(range(int(min_y), int(max_y), grid_size))

polygons = []
for x in x_coords:
    for y in y_coords:
        polygons.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))

for x in x_coords:
    for y in y_coords:
        polygons.append(Polygon([(x * grid_size, y * grid_size),
                                 ((x + 1) * grid_size, y * grid_size),
                                 ((x + 1) * grid_size, (y + 1) * grid_size),
                                 (x * grid_size, (y + 1) * grid_size)]))

# 创建 GeoDataFrame 保存网格
grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")  # 4326 是 WGS84 坐标系

# 剔除不在芝加哥边界内的网格
grid_within_chicago = gpd.overlay(grid, chicago_boundary_utm, how='intersection')

# 4. 导出裁剪后的网格为 Shapefile
grid_within_chicago.to_file("chicago_grid_100m.shp", driver="ESRI Shapefile")

print("Shapefile has been exported as 'chicago_grid_100m.shp'.")


# 2.读取 Excel 数据
# 收集所有 Excel 文件的路径
excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 逐个处理 Excel 文件
for file in tqdm(excel_files):
    # 读取 Excel 文件
    poi_data = pd.read_excel(file)

    # 将 POI 数据中的经纬度转换为地理坐标点，并检查每个 POI 属于哪个网格小方块
    # 将 POI 数据中的经纬度转换为 GeoDataFrame
    poi_gdf = gpd.GeoDataFrame(poi_data,
                               geometry=gpd.points_from_xy(poi_data.longitude, poi_data.latitude),
                               crs="EPSG:4326")

    # 将 POI 数据转换为 UTM 坐标系
    poi_gdf_utm = poi_gdf.to_crs(epsg=32616)

    # 使用 spatial join 将 POI 数据与网格匹配，找到每个 POI 属于哪个方块
    joined = gpd.sjoin(poi_gdf, grid, how="left", op="within")

    # 统计每个网格中的 POI 数量
    poi_counts = joined.groupby('index_right').size()  # 'index_right' 是网格的索引
    grid['poi_count'] = grid.index.map(poi_counts)

    # 定义输出文件名（可以使用文件名的一部分）
    output_filename = f"poi_counts_{os.path.splitext(os.path.basename(file))[0]}.csv"

    # 输出每个文件的 POI 统计结果
    grid[['geometry', 'poi_count']].to_csv(output_filename, index=False)

    print(f"Processed and saved: {output_filename}")