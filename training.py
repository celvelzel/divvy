# Python随机森林分类
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# 数据路径
path = 'data/dataset/grid_poi_counts.csv'
# 模型保存路径
model_path = 'data/model/rfc_model.pkl'

raw_data = pd.read_csv(path)  # 加载数据

# 只使用行程数量不为0的区块进行训练
train_data = raw_data[raw_data['trip_count'] != 0]

# 删除不需要的列
train_data.drop(columns=['grid_id', 'geometry'], inplace=True)

# 提取输入特征和目标变量
x = train_data.drop('trip_count', axis=1)  # 输入特征
y = train_data['trip_count']  # 目标变量

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# 训练模型
rfc = RandomForestClassifier()  # 使用默认随机森林分类器
rfc.fit(x_train, y_train)  # 使用训练数据集训练随机森林模型

# 评估模型
y_pred = rfc.predict(x_test)  # 使用分类器预测测试集的类别
importances = rfc.feature_importances_  # 计算特征重要性
print("Importances")
print(importances)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))  # 输出混淆矩阵
print("Classification Report:")
print(classification_report(y_test, y_pred))  # 输出分类报告
print("Accuracy:")  # 输出精度
print(accuracy_score(y_test, y_pred))

# 保存训练好的模型
joblib.dump(rfc, model_path)
print("模型已保存")
