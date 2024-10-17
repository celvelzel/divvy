import geopandas as gpd
from shapely.geometry import Polygon
import os
import pandas as pd

# settings
# 读取芝加哥边界的 Shapefile
chicago_boundary = gpd.read_file('data/chicago/simple_chicago.shp')

# 定义 POI 文件夹路径
folder_path = 'data/poi'  # POI Shapefile 文件夹路径


# 2. 确保所有数据使用相同的坐标系 (EPSG:4326)，再转换为 UTM 投影 (EPSG:32616)
def ensure_crs(gdf, target_crs="EPSG:4326"):
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


chicago_boundary = ensure_crs(chicago_boundary)

# 转换为 UTM 坐标系，适用于米制距离计算
chicago_boundary_utm = chicago_boundary.to_crs(epsg=3857)

# 3. 按米划分网格
grid_size = 500
min_x, min_y, max_x, max_y = chicago_boundary_utm.total_bounds

# 生成网格
polygons = []
# 通过定义每个网格四个顶点的坐标，创建一个 100 米大小的多边形（网格单元）
x_coords = list(range(int(min_x), int(max_x), grid_size))
y_coords = list(range(int(min_y), int(max_y), grid_size))

for x in x_coords:
    for y in y_coords:
        polygons.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))

# 在生成网格后
print(f"Total grid polygons created: {len(polygons)}")

# 创建一个 GeoDataFrame 保存网格
grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:3857")

# 4. 剔除不在芝加哥边界内的网格
grid_within_chicago = gpd.overlay(grid, chicago_boundary_utm, how='intersection')

# 6. 逐个处理 Shapefile 文件
output_data = pd.DataFrame({'grid_id': grid_within_chicago.index})
output_data['geometry'] = grid_within_chicago.geometry

for file in os.listdir(folder_path):
    if file.endswith('.shp'):
        file_path = os.path.join(folder_path, file)
        poi_data = gpd.read_file(file_path)
        poi_data_utm = poi_data.to_crs(epsg=3857)  # 确保转换为 UTM 坐标系

        # 使用 spatial join 将 POI 数据与网格匹配，找到每个 POI 属于哪个方块
        poi_in_grid = gpd.sjoin(poi_data_utm, grid_within_chicago, how="left", predicate="within")

        print(f"{file.split('.')[0]} Spatial join completed. Number of POIs joined: {len(poi_in_grid)}")

        # 统计每个网格中的 POI 数量
        poi_counts = poi_in_grid.groupby('index_right').size()

        # 将统计结果与网格关联，并将文件名（去掉扩展名）作为列名
        grid_within_chicago[file.split('.')[0]] = grid_within_chicago.index.map(poi_counts).fillna(0)

        # 将每个文件的统计结果添加到输出 DataFrame
        output_data[file.split('.')[0]] = grid_within_chicago[file.split('.')[0]]

# 检查输出文件夹是否存在，若不存在则创建
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 9. 导出到csv文件
output_path = os.path.join(output_dir, "grid_poi_counts.csv")
output_data.to_csv(output_path, index=False)

print(f"已成功导出到: {output_path}")

# # 8. 导出结果为 Shapefile 和 CSV 文件
# grid_within_chicago.to_file(os.path.join(output_dir, "grid_with_poi_counts.shp"), driver="ESRI Shapefile")
# grid_within_chicago[['geometry', 'poi_count']].to_csv(os.path.join(output_dir, "grid_with_poi_counts.csv"), index=False)
#
# print("每个网格的 POI 统计已成功导出。")
