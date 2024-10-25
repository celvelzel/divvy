import os
import numpy as np
import pandas as pd
import datetime
import pandas_market_calendars
from pandas_market_calendars import get_calendar

us_calendar = get_calendar('XNYS')

folder_path = 'C:\\Users\\celcelcel\\Desktop\\test1'  # 替换为存着所有你要处理的excel文件的的文件夹路径
# 获取文件夹中所有 Excel 文件的列表
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

for file in files:
    # 读取 Excel 文件
    df = pd.read_excel(os.path.join(folder_path, file))


    def check_weekday_start_time(row):

        date = pd.to_datetime(row['start_time']).strftime('%Y-%m-%d')
        if us_calendar.valid_days(start_date=date, end_date=date).size:
            return 1
        else:
            return 0

    def check_weekday_started_at(row):

        date = pd.to_datetime(row['started_at']).strftime('%Y-%m-%d')
        if us_calendar.valid_days(start_date=date, end_date=date).size:
            return 1
        else:
            return 0

    def get_hour(row):
        row['start_time_hour'] = pd.to_datetime(row['start_time']).hour


    if 'start_time' in df.columns or 'started_at' in df.columns:
        if 'start_time' in df.columns:
            df['start_time_weekday'] = df.apply(check_weekday_start_time, axis=1)
            df['start_time_hour'] = df.apply(get_hour, axis=1)
            print(df['start_time_weekday'])
        else:
            df['start_time_weekday'] = df.apply(check_weekday_started_at, axis=1)
    else:
        print("not found!!!")

    # 合并数据
    # 如果还没有创建 combined_df，则创建一个空的 DataFrame
    if 'combined_df' not in locals():
        combined_df = df
    else:
        combined_df = combined_df.append(df)

