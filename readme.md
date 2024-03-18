# Crawling station data on Divvy Bikes website

1. access

   ```
   https://account.divvybikes.com/bikesharefe-gql
   ```

2. covert raw json data to an Excel worksheet

Install python packages

```python
pip install pandas,openpyxl
```

modified the path and run divvy_json2excel.py to create a xlsx file containing information of each stations 

```python
python divvy_json2excel.py
```

3. merge the metadata from the Excel worksheet