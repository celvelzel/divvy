import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import os
from tqdm import tqdm

# 设定你的行程记录文件 csv 文件夹路径
folder_path = '../../data/prepare_counted'
# 结果输出目录
output_folder = '../../output/trip_count_week'

# 1. 划分网格
# 读取芝加哥边界的 Shapefile
chicago_boundary = gpd.read_file('../../data/chicago/simple_chicago.shp')

# 确保边界文件是 WGS84 坐标系 (EPSG:4326)
if chicago_boundary.crs != "EPSG:4326":
    chicago_boundary = chicago_boundary.to_crs("EPSG:4326")

# 转换坐标系为适合距离计算的 UTM 投影
chicago_boundary_utm = chicago_boundary.to_crs(epsg=3857)

# 使用 unary_union 合并所有多边形，保留外部边界
# 这将创建一个包含外部边界的单一多边形
chicago_boundary_tum = chicago_boundary_utm.union_all()

# 获取芝加哥的边界范围
# 返回 GeoDataFrame 中几何对象的总边界，格式为 (min_x, min_y, max_x, max_y)，分别表示几何对象的最小和最大坐标值
# 获取芝加哥边界的最小和最大 x、y 坐标，用于生成网格
min_x, min_y, max_x, max_y = chicago_boundary_utm.total_bounds

# 按米划分网格
grid_size = 500  # 单位为米

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

# 创建 GeoDataFrame 保存网格
# 使用生成的多边形创建一个包含所有网格的 GeoDataFrame
grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:3857")

print("generate grid finished")

# 将生成的网格导出为 Shapefile 以便在 QGIS 中检查
# grid.to_file("output/python_output_grid.shp", driver="ESRI Shapefile")


# 剔除不在芝加哥边界内的网格
# 用于执行两个 GeoDataFrame 之间的几何叠加操作（如交集）。该函数允许在两个几何集合之间进行空间操作
grid_within_chicago = gpd.overlay(grid, chicago_boundary_utm, how='intersection')

print("intersection finished")

# 导出裁剪后的网格为 Shapefile
# grid_within_chicago.to_file("output/python_output_grid_within_chicago.shp", driver="ESRI Shapefile")

# 2. 读取 Excel 数据
# 收集所有 Excel 文件的路径
excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 逐个处理 Excel 文件
for file in tqdm(excel_files):
    # 读取 Excel 文件
    trip_data = pd.read_csv(file)

    # 处理文件检查点
    print(f"processing file: {file}")

    if 'latitude' in trip_data.columns and 'longitude' in trip_data.columns:
        # 将 POI 数据中的经纬度转换为地理坐标点，并检查每个 POI 属于哪个网格小方块
        trip_gdf = gpd.GeoDataFrame(trip_data,
                                    geometry=gpd.points_from_xy(trip_data.longitude, trip_data.latitude),
                                    crs="EPSG:4326")
    elif 'start_lat' in trip_data.columns and 'start_lng' in trip_data.columns:
        trip_gdf = gpd.GeoDataFrame(trip_data,
                                    geometry=gpd.points_from_xy(trip_data.start_lng, trip_data.start_lat),
                                    crs="EPSG:4326")
    else:
        print(f"{file} 中没有找到经纬度列. 跳过.")
        continue

    # 将 POI 数据导出为 GeoJSON 以便在 QGIS 中加载
    # poi_gdf.to_file("output/python_output_poi_data.shp", driver="ESRI Shapefile")

    # 将 POI 数据转换为 UTM 坐标系
    trip_gdf_utm = trip_gdf.to_crs(epsg=3857)

    # 在空间连接前
    print("Starting spatial join...")
    # 使用 spatial join 将 POI 数据与网格匹配，找到每个 POI 属于哪个方块
    joined = gpd.sjoin(trip_gdf_utm, grid_within_chicago, how="left", predicate="within")
    # 在空间连接后
    print(f"Spatial join completed. Number of POIs joined: {len(joined)}")

    # 导出空间连接后的结果为 Shapefile
    # joined.to_file(f"output/{os.path.splitext(os.path.basename(file))[0]}_python_output_joined_poi_grid.shp", driver="ESRI Shapefile")

    # 统计每个网格中的 POI 数量
    trip_counts = joined['index_right'].value_counts()  # 统计网格的索引

    # 将 行程 计数填充到网格中
    grid_within_chicago['trip_count'] = grid_within_chicago.index.map(trip_counts).fillna(0)

    # 为每个网格添加目录索引
    # grid_index = grid_within_chicago.index + 1
    grid_within_chicago['grid_id'] = grid_within_chicago.index.astype(str)

    try:
        os.makedirs(output_folder)
        print('目录创建成功')
    except FileExistsError:
        print(f'目录{output_folder}已经存在')
    # 定义输出文件名
    output_filename = f"../../output/trip_count_week/trip_counts_{os.path.splitext(os.path.basename(file))[0]}.csv"

    # 输出每个文件的 行程 统计结果
    grid_within_chicago[['grid_id', 'geometry', 'trip_count']].to_csv(output_filename, index=False)

    print(f"Processed and saved: {output_filename}")
