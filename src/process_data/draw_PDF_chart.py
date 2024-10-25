import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载数据
folder_path = 'C:\\Users\\celcelcel\\Desktop\\test1'  # 替换为存着所有你要处理的excel文件的的文件夹路径
# 获取文件夹中所有 Excel 文件的列表
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

for file in files:
    # 读取 Excel 文件
    df = pd.read_excel(os.path.join(folder_path, file))

    # 转换 tripduration 列到分钟
    # 检查df是否有using time这一列
    if 'using time' in df.columns:
        df['using_time_minutes'] = df['using time'] / 60

    # 合并数据
    # 如果还没有创建 combined_df，则创建一个空的 DataFrame
    if 'combined_df' not in locals():
        combined_df = df
    else:
        combined_df = combined_df._append(df['using_time_minutes'])

data = np.array(combined_df['using_time_minutes'])

plt.hist(data, bins=1200, density=True, cumulative=False, color='steelBlue')

plt.xlabel('Time (min)')
plt.ylabel('Probability')
plt.title("Use Time Distribution: PDF")
plt.xlim(0, 120)
plt.show()