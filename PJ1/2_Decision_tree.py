import data_gainer
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

#Get data in nparrays
train_data,train_label=data_gainer.load('TrainingData.csv')
test_data,test_label=data_gainer.load('TestingData.csv')

#Create decision tree object
tree=DecisionTreeClassifier()

#train and test
tree.fit(train_data,train_label)
print('predicions of testing data:',tree.predict(test_data))

#Visualize the decision tree
plt.figure(figsize=(60, 30))
plot_tree(tree, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()