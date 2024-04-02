import datetime
import pandas as pd

now = datetime.datetime.now()
print(now)

t1 = "2019/08/01 17:12:23"
t2 = "2019/08/03 17:17:23"

using_time = (pd.to_datetime(t2) - pd.to_datetime(t1)).total_seconds()
print(using_time)
