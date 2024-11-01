import numpy as np
import matplotlib.pyplot as plt

q = np.array([-2.0, 4.0])

def f(x, P):
    C = np.exp(-2 * x[0]) + np.exp(-x[1])
    return 0.5 * x.T @ P @ x + q.T @ x + np.log(C)

def gradient(x, P):
    C = np.exp(-2 * x[0]) + np.exp(-x[1])
    B = np.array([2 * np.exp(-2 * x[0]), np.exp(-x[1])])
    return P @ x + q - (1 / C) * B

def gradient_descent_exact_line_search(x_init, P, learning_rate=0.01, tolerance=0.01):
    x_values = [x_init]
    y_values = [f(x_init, P)]
    gradi = gradient(x_init, P)

    while np.linalg.norm(gradi) >= tolerance:
        x_new = x_values[-1] - learning_rate * gradi
        x_values.append(x_new)
        y_values.append(f(x_new, P))
        gradi = gradient(x_new, P)

    return x_values, y_values

def gradient_descent_backtracking_line_search(x_init, P, alpha=0.15, gama=0.7, beta=0.8, tolerance=0.01):
    x_values = [x_init]
    y_values = [f(x_init, P)]
    gradi = gradient(x_init, P)

    while np.linalg.norm(gradi) >= tolerance:
        if f(x_values[-1] - alpha * np.array([1, 1]), P) > f(x_values[-1], P) + np.linalg.norm(gradi) * gama * alpha:
            alpha *= beta
        x_new = x_values[-1] - alpha * gradi
        x_values.append(x_new)
        y_values.append(f(x_new, P))
        gradi = gradient(x_new, P)

    return x_values, y_values

def plot_results(x_values, y_values, P):
    x1_values = [xi[0] for xi in x_values]
    x2_values = [xi[1] for xi in x_values]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x1_grid = np.linspace(min(x1_values) - 1, max(x1_values) + 1, 100)
    x2_grid = np.linspace(min(x2_values) - 1, max(x2_values) + 1, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    F = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_grid = np.array([X1[i, j], X2[i, j]])
            F[i, j] = f(x_grid, P)

    ax.plot_surface(X1, X2, F, alpha=0.6, cmap='viridis', edgecolor='none')
    ax.plot(x1_values[:-1], x2_values[:-1], y_values[:-1], 'o-', color='red')
    ax.scatter(x1_values[-1], x2_values[-1], y_values[-1], c='gold', marker='*', s=200, edgecolors='black', label='final point')

    ax.view_init(elev=30, azim=45)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_zlabel('$f(x)$', fontsize=12)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    x_0 = np.array([1.0, 2.0])

    x_values, y_values = gradient_descent_exact_line_search(x_0, P=np.array([[3, 4], [4, 6]]))
    print('1.(a)x=', x_values[-1])
    plot_results(x_values, y_values, P=np.array([[3, 4], [4, 6]]))

    x_values, y_values = gradient_descent_backtracking_line_search(x_0, P=np.array([[3, 4], [4, 6]]))
    print('1.(b)x=', x_values[-1])
    plot_results(x_values, y_values, P=np.array([[3, 4], [4, 6]]))

    P = np.array([[5.005, 4.995], [4.995, 5.005]])
    x_values, y_values = gradient_descent_exact_line_search(x_0, P)
    print('2.(a)x=', x_values[-1])
    plot_results(x_values, y_values, P)

    x_values, y_values = gradient_descent_backtracking_line_search(x_0, P)
    print('2.(b)x=', x_values[-1])
    plot_results(x_values, y_values, P)
