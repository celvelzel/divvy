import joblib
import pandas as pd

# 加载随机森林模型
model = joblib.load('../../model/rfr_model_electric.pkl')

# 检查模型是否支持特征重要性
if hasattr(model, 'feature_importances_'):
    # 获取特征重要性
    feature_importance = model.feature_importances_
    print("Feature Importance:", feature_importance)
else:
    raise ValueError("The loaded model does not support feature importance.")

# 读取 Excel 文件以获取特征名称
csv_file = '../../data/dataset/grid_poi_counts.csv'  # 替换为你的csv文件路径
df = pd.read_csv(csv_file)
df.drop(columns=['grid_id'], inplace=True, errors='ignore')
df.drop(columns=['geometry'], inplace=True, errors='ignore')

# 获取首行作为特征名称
feature_names = df.columns.tolist()

feature_names.insert(0, '月份')
feature_names.insert(1, '一年的第几天')

# 确保特征名称的数量与特征重要性的数量一致
if len(feature_names) != len(feature_importance):
    raise ValueError("The number of feature names does not match the number of features in the model.")

# 将特征重要性与特征名称结合
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance * 100
})

# 按重要性排序（从高到低）
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 打印结果
print("Feature Importance with Names:")
print(importance_df)

