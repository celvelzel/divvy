import pandas as pd

# 读取第一个Excel表格
df1 = pd.read_excel(".xlsx")

# 读取第二个Excel表格
df2 = pd.read_excel("second_excel.xlsx")

# 将第一个表格中的'totalDocks'列根据'stationName'合并到第二个表格
df_merged = df2.merge(df1[['stationName', 'totalDocks']], on='stationName', how='left')

# 将合并后的结果写入新的Excel文件
df_merged.to_excel("merged_excel.xlsx", index=False)
