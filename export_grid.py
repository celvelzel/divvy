import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import os
from tqdm import tqdm

# 设定你的行程记录文件 csv 文件夹路径
folder_path = 'data/grid'

# 1. 划分网格
# 读取芝加哥边界的 Shapefile
chicago_boundary = gpd.read_file('data/chicago/simple_chicago.shp')

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
# print(f"Total grid polygons created: {len(polygons)}")

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
os.mkdir('output/grid')
grid_within_chicago.to_file("output/grid/python_output_grid_within_chicago.shp", driver="ESRI Shapefile")
