import pandas as pd

def load(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Separate the last column as the label
    labels = data.iloc[:, -1]

    # Remaining columns as data
    data = data.iloc[:, :-1]
    data_np = data.to_numpy()
    labels_np = labels.to_numpy()

    return data_np, labels_np
