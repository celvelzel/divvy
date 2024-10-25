
import pandas as pd

# 加载 Excel 文件
file_path = 'C:\\Users\\celcelcel\\Desktop\\2019.8_s.xlsx'  # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path)


# 1. 创建一个函数来检查 non-using time 是否超过特定时间
def check_time_24(row):
    if row['non-using time'] > (24 * 3600):
        return 0
    else:
        return row['non-using time']

def check_time_48(row):
    if row['non-using time'] > (48 * 3600):
        return 0
    else:
        return row['non-using time']

# 2. 应用函数到每一行
df['non-using time<24hrs'] = df.apply(check_time_24, axis=1)
df['non-using time<48hrs'] = df.apply(check_time_48, axis=1)

print("check finished")

# 保存更新后的数据框到新的 Excel 文件
output_file_path = 'C:\\Users\\celcelcel\\Desktop\\2019.8_final.xlsx'  # 替换为你希望保存新文件的路径
df.to_excel(output_file_path, index=False)
