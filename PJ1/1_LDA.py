import data_gainer
import sklearn.discriminant_analysis
import matplotlib.pyplot as plt

#Get data in nparrays
train_data,train_label=data_gainer.load('TrainingData.csv')
test_data,test_label=data_gainer.load('TestingData.csv')

#Train LDA
LDA=sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
LDA.fit(train_data,train_label)
w=LDA.coef_.T

#Get projections of data
y_k=test_data@w # y_k = wT*xk

#Get values of error 1 and error 2 when threshold is varying
#initializing varibles
Error_1=[]
Error_2=[]
Thres=[]
threshold=min(y_k)
best_threshold=threshold
error_sum=1
#varying threshold for the tiniest y to the biggest
while(threshold<-max(y_k)):
    error1 = 0
    error2 = 0
    for i, y in enumerate(y_k):
        if y <= threshold:
            prediction = 0
        else:
            prediction = 1
        if test_label[i] == 0 and prediction == 1:
            error1 += 1
        elif test_label[i] == 1 and prediction == 0:
            error2 += 1
    error1 /= (i + 1)
    error2 /= (i + 1)
    error_sum_temp=error1+error2
    # Get the best threshold when the sum of error 1 and error 2 is minimized
    if error_sum_temp < error_sum:
        error_sum=error_sum_temp
        best_threshold=threshold.copy()
    Error_1.append(error1)
    Error_2.append(error2)
    Thres.append(threshold.copy())
    threshold+=0.01

#Draw plot for the variation of error 1 and error 2 with threshold
plt.plot(Thres, Error_1, label="Type 1 Error Rate")
plt.plot(Thres, Error_2, label="Type 2 Error Rate")
plt.xlabel("Threshold")
plt.ylabel("Error Rate")
plt.legend()
plt.title("Error Rates vs. Threshold in LDA")
plt.show()

#making predictions based on the best threshold and projections y_k
Predictions=[]
for y in (y_k):
    if y <= best_threshold:
        Predictions.append(0)
    else:
        Predictions.append(1)
print(Predictions)
print(test_label)


