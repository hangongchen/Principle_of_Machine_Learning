from PJ1 import data_gainer
from sklearn import neighbors
import matplotlib.pyplot as plt

test_data, test_label = data_gainer.load('TestingData.csv')


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

if __name__ == "__main__":
    # 仅在直接运行 KNN.py 时执行的代码
    # 测试代码或其他逻辑
    # Get data in nparrays
    train_data, train_label = data_gainer.load('TrainingData.csv')
    test_data, test_label = data_gainer.load('TestingData.csv')
    # Create KNN object
    KNN1 = neighbors.KNeighborsClassifier(n_neighbors=1)
    KNN3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    KNN5 = neighbors.KNeighborsClassifier(n_neighbors=5)
    KNN10 = neighbors.KNeighborsClassifier(n_neighbors=10)

    # train and test
    KNN1.fit(train_data, train_label)
    KNN3.fit(train_data, train_label)
    KNN5.fit(train_data, train_label)
    KNN10.fit(train_data, train_label)

    print('KNN1.predict:', KNN1.predict(test_data))
    print('KNN3.predict:', KNN3.predict(test_data))
    print('KNN5.predict:', KNN5.predict(test_data))
    print('KNN10.predict:', KNN10.predict(test_data))
    # Create a function to get error1 and error2

    # calculate error1 and 2 for each knn
    Error1 = []
    Error2 = []
    error1, error2 = get_error(KNN1.predict(test_data))
    Error1.append(error1)
    Error2.append(error2)
    error1, error2 = get_error(KNN3.predict(test_data))
    Error1.append(error1)
    Error2.append(error2)
    error1, error2 = get_error(KNN5.predict(test_data))
    Error1.append(error1)
    Error2.append(error2)
    error1, error2 = get_error(KNN10.predict(test_data))
    Error1.append(error1)
    Error2.append(error2)

    # visulization
    k = [1, 3, 5, 10]
    plt.scatter(k, Error1, label='error1')
    plt.scatter(k, Error2, label='error2')
    plt.plot(k, [a + b for a, b in zip(Error1, Error2)], label='sum error', color='g')
    plt.xlabel("k")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.title("Error Rates vs. k")
    plt.show()