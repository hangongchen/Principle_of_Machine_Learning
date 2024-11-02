import numpy as np
from scipy.signal import correlate2d

# Define the input matrix with zero padding
input_matrix = np.array([
    [1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0]
])

# Define the kernel
kernel = np.array([
    [2, 1, 2],
    [0, 2, 0],
    [1, 2, 1]
])

# Perform convolution with zero padding
convolution_result = correlate2d(input_matrix, kernel, mode='same')

# Perform max pooling with 2x2 kernel and stride 2
def max_pooling(matrix, pool_size=2, stride=2):
    output_shape = (
        (matrix.shape[0] - pool_size) // stride + 1,
        (matrix.shape[1] - pool_size) // stride + 1
    )
    pooled_matrix = np.zeros(output_shape)
    for i in range(0, matrix.shape[0] - pool_size + 1, stride):
        for j in range(0, matrix.shape[1] - pool_size + 1, stride):
            pooled_matrix[i // stride, j // stride] = np.max(matrix[i:i + pool_size, j:j + pool_size])
    return pooled_matrix

# Apply max pooling to the convolution result
max_pooling_result = max_pooling(convolution_result)
print('convolution_result:\n',convolution_result)
print('max_pooling_result:\n',max_pooling_result)
