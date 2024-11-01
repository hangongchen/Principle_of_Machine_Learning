from PJ1 import data_gainer
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#Get data in nparrays
train_data,train_label=data_gainer.load('TrainingData.csv')
test_data,test_label=data_gainer.load('TestingData.csv')
def get_error(predictions,labels=test_label):
    error1=0
    error2=0
    for i, prediction in enumerate(predictions):
        if labels[i] == 0 and prediction == 1:
            error1 += 1
        elif labels[i] == 1 and prediction == 0:
            error2 += 1
    error1 /= (i + 1)
    error2 /= (i + 1)
    return error1,error2
#For better visulization, I select k from 1 to 15
#initialize varibles
K=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
PCAs=[]
train_data_pca = []
test_data_pca = []
KNNerror1=[]
KNNerror2=[]
SVMerror1=[]
SVMerror2=[]
#apply PCA, KNN and SVM throughout k
for i,k in enumerate(K):
    #Applying PCA
    PCAs.append(PCA(n_components=k))
    PCAs[i].fit(train_data,train_label)
    train_data_pca.append(PCAs[i].transform(train_data))
    test_data_pca.append(PCAs[i].transform(test_data))
    #Applying KNN which k(for knn)=5
    KNN5=neighbors.KNeighborsClassifier(n_neighbors=5)
    KNN5.fit(train_data_pca[i],train_label)
    error1, error2 = get_error(KNN5.predict(test_data_pca[i]))
    KNNerror1.append(error1)
    KNNerror2.append(error2)
    #Applying SVM with kernel function RBF
    svc = SVC(kernel='rbf')
    svc.fit(train_data_pca[i], train_label)
    error1, error2 = get_error(svc.predict(test_data_pca[i]))
    SVMerror1.append(error1)
    SVMerror2.append(error2)
#calculate total error
KNNerror_sum = [e1 + e2 for e1, e2 in zip(KNNerror1, KNNerror2)]
SVMerror_sum = [e1 + e2 for e1, e2 in zip(SVMerror1, SVMerror2)]

#visualization for KNN
plt.figure(figsize=(10, 6))
plt.plot(K, KNNerror1, marker='o', label="KNN Error 1")
plt.plot(K, KNNerror2, marker='o', label="KNN Error 2")
plt.plot(K, KNNerror_sum, marker='o', linestyle='--', label="KNN Error Sum")
plt.xlabel("Number of Principal Components (K)")
plt.ylabel("Error")
plt.title("KNN Errors and Sum vs. Number of Principal Components")
plt.legend()
plt.grid(True)
plt.show()

#visualization for PCA
plt.figure(figsize=(10, 6))
plt.plot(K, SVMerror1, marker='o', label="SVM Error 1")
plt.plot(K, SVMerror2, marker='o', label="SVM Error 2")
plt.plot(K, SVMerror_sum, marker='o', linestyle='--', label="SVM Error Sum")
plt.xlabel("Number of Principal Components (K)")
plt.ylabel("Error")
plt.title("SVM Errors and Sum vs. Number of Principal Components")
plt.legend()
plt.grid(True)
plt.show()




