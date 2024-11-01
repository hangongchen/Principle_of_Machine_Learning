import pandas as pd
from sklearn.linear_model import Perceptron
import numpy as np

data = pd.read_csv('Pizza.csv')

X = data.iloc[:, 2:].values
y = data.iloc[:, 0].values

label_mapping = {'A': 0, 'B': 1, 'C': 2}
mask = [label in label_mapping for label in y]
X = X[mask]
y = np.array([label_mapping[label] for label in y if label in label_mapping])

classifiers = {}
pairs = [(0, 1), (0, 2), (1, 2)]

for (i, j) in pairs:
    mask = (y == i) | (y == j)
    X_pair = X[mask]
    y_pair = y[mask]

    # 修改 y_pair 为 1 和 -1
    y_pair = np.where(y_pair == i, 1, -1)

    clf = Perceptron(max_iter=1000, tol=1e-3)
    clf.fit(X_pair, y_pair)
    classifiers[(i, j)] = clf

    w = clf.coef_[0]
    b = clf.intercept_[0]
    print(f"Classifier for brands {list(label_mapping.keys())[i]} and {list(label_mapping.keys())[j]}:")
    print(f"Hyperplane equation: {w[0]}*x1 + {w[1]}*x2 + {w[2]}*x3 + {w[3]}*x4 + {w[4]}*x5 + {w[5]}*x6 + {w[6]}*x7 + {b} = 0\n")

margins = {}

for (i, j), clf in classifiers.items():
    w = clf.coef_[0]
    norm_w = np.linalg.norm(w)
    margin = 1 / norm_w
    margins[(i, j)] = margin
    print(f"Margin for brands {list(label_mapping.keys())[i]} and {list(label_mapping.keys())[j]}: {margin}\n")

samples = np.array([
    [49.29, 24.82, 21.68, 2.76, 0.52, 1.47, 3.00],
    [30.95, 19.81, 42.28, 5.11, 1.67, 1.85, 4.67],
    [50.33, 13.28, 28.43, 3.58, 1.03, 4.38, 3.27]
])

def classify_sample(sample):
    votes = {0: 0, 1: 0, 2: 0}
    for (i, j), clf in classifiers.items():
        prediction = clf.predict([sample])[0]
        if prediction == 1:
            votes[i] += 1
        else:
            votes[j] += 1
    return max(votes, key=votes.get)

for idx, sample in enumerate(samples, start=1):
    label_idx = classify_sample(sample)
    label = [key for key, value in label_mapping.items() if value == label_idx][0]
    print(f"s{idx} is classified as brand '{label}'")
