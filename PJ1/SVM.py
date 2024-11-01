import data_gainer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#Get data in nparrays
train_data,train_label=data_gainer.load('TrainingData.csv')
test_data,test_label=data_gainer.load('TestingData.csv')

#Train SVM and predict for testing set
SVC = SVC(kernel='rbf')
SVC.fit(train_data, train_label)
print('SVM Predictions:', SVC.predict(test_data))
image = plt.imread('kernel.png')
plt.imshow(image)
plt.title("kernel function")
plt.axis('off')
plt.show()