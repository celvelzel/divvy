import datetime
import pandas as pd

now = datetime.datetime.now()
print(now)

t1 = "2019-09-12 13:10:21"
t2 = "2019/08/03 17:17:23"

print(pd.to_datetime(t1))
using_time = (pd.to_datetime(t2) - pd.to_datetime(t1)).total_seconds()
print(using_time)


