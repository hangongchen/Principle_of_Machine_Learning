import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


# 加载数据
file_path = 'HW9/moonDataset.csv'
moon_data = pd.read_csv(file_path)

# 数据概览
print("数据结构：")
print(moon_data.head())
print(f"数据集大小: {moon_data.shape}")

# 划分训练集和测试集
train_data = moon_data.iloc[:150]
test_data = moon_data.iloc[150:]
print(test_data)

# 创建 50 个 bootstrap 训练数据集
bootstrapped_datasets = [train_data.sample(n=150, replace=True, random_state=i) for i in range(50)]

# 打印第一个 bootstrap 数据集以验证
print("\n第一个 bootstrap 数据集：")
print(bootstrapped_datasets[0])
classifier = MLPClassifier(hidden_layer_sizes=(10,), random_state=1, max_iter=500)
error_rate=[]
for dataset in bootstrapped_datasets:
    train_x= dataset.iloc[:,:-1]
    train_y= dataset.iloc[:,-1]
    classifier.fit(train_x,train_y)
    pred=classifier.predict(test_data.iloc[:,:-1])
    error_rate.append(1-accuracy_score(pred,test_data.iloc[:,-1]))
# 改成折线图
plt.figure(figsize=(10, 6))
plt.scatter(range(1, 51), error_rate, marker='o', label="Error Rate per Network")
plt.title("Line Plot of Error Rates (50 Neural Networks)")
plt.xlabel("Bootstrap Dataset Index")
plt.ylabel("Error Rate")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()


error_rate2=[]
ensemble_sizes=[5,10,15,20]
for ensemble_size in ensemble_sizes:
    classifier2 = BaggingClassifier(estimator=classifier,n_estimators=ensemble_size,random_state=1)
    classifier2.fit(train_x,train_y)
    pred=classifier2.predict(test_data.iloc[:,:-1])
    error_rate2.append(1-accuracy_score(pred,test_data.iloc[:,-1]))

plt.figure(figsize=(10, 6))
plt.plot(ensemble_sizes, error_rate2, marker='o', linestyle='-', label="Error Rate")
plt.title("Error Rate vs Ensemble Size (Bagging Classifier)")
plt.xlabel("Ensemble Size (m)")
plt.ylabel("Error Rate")
plt.xticks(ensemble_sizes)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()