import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
numpy.random.seed(100)

def gen_data(clusterA_mid_1, Cluster_A_size_1, clusterA_mid_2, Cluster_A_size_2, cluster_B_mid, cluster_B_size):
    # Generating class A with two clusters
    classA = np.concatenate((
        np.random.randn(10, 2) * Cluster_A_size_1 + clusterA_mid_1,
        np.random.randn(10, 2) * Cluster_A_size_2 + clusterA_mid_2))

    # Generating class B with one cluster
    classB = np.random.randn(20, 2) * cluster_B_size + cluster_B_mid
    # Concatenating inputs
    inputs = np.concatenate((classA, classB))
    # Creating target labels (1 for class A, -1 for class B)
    targets = np.concatenate((
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    ))
    # Number of samples
    N = inputs.shape[0]

    # Permuting the dataset
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    alpha = numpy.zeros(N)

    return alpha, targets, inputs, N, classA, classB


alpha, targets, inputs, N, classA, classB = gen_data(
    clusterA_mid_1=[1.5, 0.5], Cluster_A_size_1=0.2,
    clusterA_mid_2=[-1.5, 0.5], Cluster_A_size_2=0.2,
    cluster_B_mid=[0.0, -0.5], cluster_B_size=0.2)


# bounds= [(0, C) for b in range(N)]
bounds = [(0, None) for b in range(N)]


def kernels(x, y, type, p=2, sigma=1.0):
    def linear_kernel(x, y):
        """Computes the Linear Kernel between vectors x and y."""
        return np.dot(x, y)

    def polynomial_kernel(x, y, p=2):
        """Computes the Polynomial Kernel with degree p."""
        return (np.dot(x, y) + 1) ** p

    def rbf_kernel(x, y, sigma=1.0):
        """Computes the Radial Basis Function (RBF) Kernel with parameter sigma."""
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

    if type == 'L':
        return linear_kernel(x, y)
    elif type == 'P':
        return polynomial_kernel(x, y, p=p)
    elif type == 'R':
        return rbf_kernel(x, y, sigma)


def calc_P(t, X, kernel_function, **kernel_params):
    n_samples = X.shape[0]
    # Compute the Kernel matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_function(X[i], X[j], **kernel_params)
    # Compute the Lagrange dual function value
    P = t[:, None] * t[None, :] * K
    return P


P = calc_P(targets, inputs, kernels, type='L', p=2, sigma=1.0)


def objective(alpha):
    term1 = np.sum(alpha)
    term2 = 0.5 * np.sum(alpha[:, None] * alpha[None, :] * P)
    return term2 - term1


def zerofun(alpha):
    return np.dot(alpha, targets)


def calc_b(S, alphas, targets, kernels, inputs, **kernel_params):
    b = 0
    for i in range(N):
        b += alphas[i] * targets[i] * kernels(S[1], inputs[i], **kernel_params)
    return b - S[2]


def ind_func(S, alphas, targets, kernels, inputs, b, **kernel_params):
    ind = 0
    for i in range(N):
        ind += alphas[i] * targets[i] * kernels(S, inputs[i], **kernel_params)
    return ind - b


XC = {'type': 'eq', 'fun': zerofun}
ret = minimize(objective, alpha, bounds=bounds, constraints=XC)
alpha = ret['x']
Non_zero_alpha = [[alpha[idx], inputs[idx], targets[idx], idx] for idx in range(N) if alpha[idx] > 10 ** (-5)]
S = Non_zero_alpha[0]
print(Non_zero_alpha)

b = calc_b(S, alpha, targets, kernels, inputs, type='L', p=2, sigma=1.0)
print(b)
# Define grid
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array(
    [[ind_func((x, y), alpha, targets, kernels, inputs, b, type='L', p=2, sigma=1.0) for x in xgrid] for y in ygrid])

# Create contour plot

# Assuming classA and classB are lists of (x, y) coordinate tuples
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

plt.axis('equal')  # Force same scale on both axes

plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

plt.show()
plt.savefig('svmplot.pdf')  # Save a copy in a file

