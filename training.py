# Python随机森林分类
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

path = 'data/dataset/grid_poi_counts.csv'  # 数据路径
rawdata = pd.read_csv(path)  # 加载数据
rawdata.drop(columns=['grid_id', 'geometry'], inplace=True)
x = rawdata.drop('trip_count', axis=1)  # 输入特征
y = rawdata['trip_count']  # 目标变量
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # 30%为测试集，则70%为训练集
rfc = RandomForestClassifier()  # 使用默认随机森林分类器
rfc.fit(x_train, y_train)  # 使用训练数据集训练随机森林模型
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
