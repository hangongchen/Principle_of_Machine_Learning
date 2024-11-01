import pandas as pd
import numpy as np

file_path = 'Pizza.csv'
pizza_data = pd.read_csv(file_path)

features = pizza_data[['mois', 'prot', 'fat', 'ash', 'sodium', 'carb', 'cal']]

mean_vector = features.mean()
print("Mean Feature Vector:")
print(mean_vector)

scatter_matrix = np.cov(features, rowvar=False)
print("\nScatter Matrix (Before Standardization):")
print(scatter_matrix)

eigenvalues, eigenvectors = np.linalg.eig(scatter_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

w1 = eigenvectors_sorted[:, 0] / np.linalg.norm(eigenvectors_sorted[:, 0])
w2 = eigenvectors_sorted[:, 1] / np.linalg.norm(eigenvectors_sorted[:, 1])

print("\nFirst Principal Component (w1) Before Standardization:")
print(w1)

print("\nSecond Principal Component (w2) Before Standardization:")
print(w2)

a1 = np.dot(features, w1)
a2 = np.dot(features, w2)

approx_data = mean_vector.values + np.outer(a1, w1) + np.outer(a2, w2)

original_first_two = features.iloc[:2]
approximated_first_two = approx_data[:2]

comparison_df = pd.DataFrame({
    'Original Moisture': original_first_two['mois'],
    'Approximated Moisture': approximated_first_two[:, 0],
    'Original Protein': original_first_two['prot'],
    'Approximated Protein': approximated_first_two[:, 1],
    'Original Fat': original_first_two['fat'],
    'Approximated Fat': approximated_first_two[:, 2],
    'Original Ash': original_first_two['ash'],
    'Approximated Ash': approximated_first_two[:, 3],
    'Original Sodium': original_first_two['sodium'],
    'Approximated Sodium': approximated_first_two[:, 4],
    'Original Carb': original_first_two['carb'],
    'Approximated Carb': approximated_first_two[:, 5],
    'Original Calories': original_first_two['cal'],
    'Approximated Calories': approximated_first_two[:, 6],
})

print("\nComparison of Original and Approximated Feature Values for the First Two Samples (Before Standardization):")
print(comparison_df.to_string())

std_vector = features.std()

standardized_data = (features - mean_vector) / std_vector

scatter_matrix_standardized = np.cov(standardized_data, rowvar=False)

print("\nScatter Matrix (After Standardization):")
print(scatter_matrix_standardized)

eigenvalues_standardized, eigenvectors_standardized = np.linalg.eig(scatter_matrix_standardized)

sorted_indices_standardized = np.argsort(eigenvalues_standardized)[::-1]
eigenvalues_sorted_standardized = eigenvalues_standardized[sorted_indices_standardized]
eigenvectors_sorted_standardized = eigenvectors_standardized[:, sorted_indices_standardized]

w1_standardized = eigenvectors_sorted_standardized[:, 0] / np.linalg.norm(eigenvectors_sorted_standardized[:, 0])
w2_standardized = eigenvectors_sorted_standardized[:, 1] / np.linalg.norm(eigenvectors_sorted_standardized[:, 1])

print("\nFirst Principal Component (w1) After Standardization:")
print(w1_standardized)

print("\nSecond Principal Component (w2) After Standardization:")
print(w2_standardized)

a1_standardized = np.dot(standardized_data, w1_standardized)
a2_standardized = np.dot(standardized_data, w2_standardized)

approx_data_standardized = mean_vector.values + np.outer(a1_standardized, w1_standardized) + np.outer(a2_standardized, w2_standardized)

approx_data_standardized_original_scale = approx_data_standardized * std_vector.values + mean_vector.values

original_first_two_standardized = features.iloc[:2]
approximated_first_two_standardized = approx_data_standardized_original_scale[:2]

comparison_df_standardized = pd.DataFrame({
    'Original Moisture': original_first_two_standardized['mois'],
    'Approximated Moisture': approximated_first_two_standardized[:, 0],
    'Original Protein': original_first_two_standardized['prot'],
    'Approximated Protein': approximated_first_two_standardized[:, 1],
    'Original Fat': original_first_two_standardized['fat'],
    'Approximated Fat': approximated_first_two_standardized[:, 2],
    'Original Ash': original_first_two_standardized['ash'],
    'Approximated Ash': approximated_first_two_standardized[:, 3],
    'Original Sodium': original_first_two_standardized['sodium'],
    'Approximated Sodium': approximated_first_two_standardized[:, 4],
    'Original Carb': original_first_two_standardized['carb'],
    'Approximated Carb': approximated_first_two_standardized[:, 5],
    'Original Calories': original_first_two_standardized['cal'],
    'Approximated Calories': approximated_first_two_standardized[:, 6],
})

print("\nComparison of Original and Approximated Feature Values for the First Two Samples (After Standardization):")
print(comparison_df_standardized.to_string())
