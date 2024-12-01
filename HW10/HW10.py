# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Load the data
data = pd.read_csv('Pizza.csv')

# Select columns 3 through 9 (assuming zero-based indexing)
# Adjust the column indices if necessary
X = data.iloc[:, 2:9].values  # Columns 3 to 9

# Number of samples
n_samples = X.shape[0]

# Standardize the data (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Part (a): Optimal Linear Autoencoder using PCA
mse_pca = []

for h in range(1, 7):
    # Perform PCA with h components
    pca = PCA(n_components=h)
    Z = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(Z)

    # Compute MSE
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    mse_pca.append(mse)

# Plot MSE vs. Code Size h for PCA
plt.figure(figsize=(10, 5))
plt.plot(range(1, 7), mse_pca, marker='o', label='Linear Autoencoder (PCA)')
plt.title('MSE vs. Code Size h (Linear Autoencoder)')
plt.xlabel('Code Size h')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()
plt.show()

# Part (b): Autoencoder with ReLU Activation Function
mse_relu = []


# Define a function to create and train the autoencoder
def train_autoencoder(h, X_train):
    input_dim = X_train.shape[1]

    # Define the autoencoder model
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(h, activation='relu')(input_layer)
    # Decoder
    decoded = Dense(input_dim, activation='linear')(encoded)

    # Compile the model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    # Train the model
    autoencoder.fit(X_train, X_train,
                    epochs=100,
                    batch_size=16,
                    verbose=0)
    return autoencoder


for h in range(1, 7):
    # Train autoencoder
    autoencoder = train_autoencoder(h, X_scaled)
    # Reconstruct the data
    X_reconstructed = autoencoder.predict(X_scaled)
    # Compute MSE
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    mse_relu.append(mse)

# Plot MSE vs. Code Size h for ReLU Autoencoder
plt.figure(figsize=(10, 5))
plt.plot(range(1, 7), mse_relu, marker='o', color='orange', label='ReLU Autoencoder')
plt.title('MSE vs. Code Size h (ReLU Autoencoder)')
plt.xlabel('Code Size h')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()
plt.show()

# Combined Plot for Comparison
plt.figure(figsize=(10, 5))
plt.plot(range(1, 7), mse_pca, marker='o', label='Linear Autoencoder (PCA)')
plt.plot(range(1, 7), mse_relu, marker='o', color='orange', label='ReLU Autoencoder')
plt.title('MSE vs. Code Size h')
plt.xlabel('Code Size h')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()
plt.show()
