import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np

file_path = 'DiabetesTraining.csv'
data = pd.read_csv(file_path)

features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X = data[features]
y = data['diabetes']

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

lda = LDA(n_components=1)
lda.fit(X_standardized, y)

w_vector = lda.coef_

def gini_impurity(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = (group.count(class_val)) / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / n_instances)
    return gini

def entropy(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    entropy = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = (group.count(class_val)) / size
            if proportion > 0:
                score += -proportion * np.log2(proportion)
        entropy += (score) * (size / n_instances)
    return entropy

def split_by_attribute(attribute):
    group_0 = data[data[attribute] == 0]['diabetes'].tolist()
    group_1 = data[data[attribute] == 1]['diabetes'].tolist()
    return [group_0, group_1]

attributes = ['hypertension', 'heart_disease']
classes = [0, 1]

results = {}
for attribute in attributes:
    groups = split_by_attribute(attribute)
    gini = gini_impurity(groups, classes)
    initial_entropy = entropy([data['diabetes'].tolist()], classes)
    post_split_entropy = entropy(groups, classes)
    info_gain = initial_entropy - post_split_entropy
    results[attribute] = {
        'Gini Impurity': gini,
        'Information Gain': info_gain
    }

print('w=',w_vector)
print(results)
