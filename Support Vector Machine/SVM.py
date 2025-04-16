import numpy as np
import random
import math

import numpy.random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

numpy.random.seed(100)
class SVMClassifier:
    def __init__(self, clusterA_mid_1, Cluster_A_size_1, clusterA_mid_2, Cluster_A_size_2,
                 cluster_B_mid, cluster_B_size, kernel_type='L', p=2, sigma=1.0,C=None):
        self.kernel_type = kernel_type
        self.p = p
        self.sigma = sigma

        # Generate dataset
        self.alpha, self.targets, self.inputs, self.N, self.classA, self.classB = self.gen_data(
            clusterA_mid_1, Cluster_A_size_1, clusterA_mid_2, Cluster_A_size_2, cluster_B_mid, cluster_B_size)

        self.bounds = [(0, C) for _ in range(self.N)]

        # Compute P matrix
        self.P = self.calc_P()

        # Optimize
        self.alpha = self.optimize()

        # Extract support vectors
        self.support_vectors = self.get_support_vectors()
        print(self.support_vectors)

        # Compute bias term b
        self.b = self.calc_b()
        print(self.b)


    def gen_data(self, clusterA_mid_1, Cluster_A_size_1, clusterA_mid_2, Cluster_A_size_2, cluster_B_mid,
                 cluster_B_size):
        classA = np.concatenate((
            np.random.randn(10, 2) * Cluster_A_size_1 + clusterA_mid_1,
            np.random.randn(10, 2) * Cluster_A_size_2 + clusterA_mid_2))
        classB = np.random.randn(20, 2) * cluster_B_size + cluster_B_mid
        inputs = np.concatenate((classA, classB))
        targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

        N = inputs.shape[0]
        permute = list(range(N))
        random.shuffle(permute)

        return np.zeros(N), targets[permute], inputs[permute, :], N, classA, classB

    def kernels(self, x, y):
        if self.kernel_type == 'L':
            return np.dot(x, y)
        elif self.kernel_type == 'P':
            return (np.dot(x, y) + 1) ** self.p
        elif self.kernel_type == 'R':
            return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def calc_P(self):
        K = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                K[i, j] = self.kernels(self.inputs[i], self.inputs[j])
        return self.targets[:, None] * self.targets[None, :] * K

    def objective(self, alpha):
        return 0.5 * np.sum(alpha[:, None] * alpha[None, :] * self.P) - np.sum(alpha)

    def zerofun(self, alpha):
        return np.dot(alpha, self.targets)

    def optimize(self):
        XC = {'type': 'eq', 'fun': self.zerofun}
        ret = minimize(self.objective, self.alpha, bounds=self.bounds, constraints=XC)
        return ret['x']

    def get_support_vectors(self):
        return [[self.alpha[idx], self.inputs[idx], self.targets[idx]]
                for idx in range(self.N) if self.alpha[idx] > 1e-5]

    def calc_b(self):
        S = self.support_vectors[0]
        return sum(self.alpha[i] * self.targets[i] * self.kernels(S[1], self.inputs[i])
                   for i in range(self.N)) - S[2]

    def ind_func(self, S):
        return sum(self.alpha[i] * self.targets[i] * self.kernels(S, self.inputs[i])
                   for i in range(self.N)) - self.b

    def plot(self):
        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-4, 4)
        grid = np.array([[self.ind_func((x, y)) for x in xgrid] for y in ygrid])

        plt.plot([p[0] for p in self.classA], [p[1] for p in self.classA], 'b.')
        plt.plot([p[0] for p in self.classB], [p[1] for p in self.classB], 'r.')
        plt.axis('equal')
        plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
        plt.show()


# Example usage with different parameters
def Q_1():
    for i in range(5):
        svm1 = SVMClassifier([1.5, 0.5], 0.2, [-1.5, 0.5-i/5], 0.2, [0.0, -0.5], 0.2, kernel_type='L')
        svm1.plot()

    for i in range(5):
        svm1 = SVMClassifier([1.5, 0.5], 0.2, [-1.5, 0.5], 0.2+i/5, [0.0, -0.5], 0.2, kernel_type='L')
        svm1.plot()

def Q_2():
    for i in range(5):
        svm1 = SVMClassifier([1.5, 0.5], 0.2, [-1.5, 0.5-i/5], 0.2, [0.0, -0.5], 0.2, kernel_type='P', p=3)
        svm1.plot()

    for i in range(5):
        svm1 = SVMClassifier([1.5, 0.5], 0.2, [-1.5, 0.5], 0.2 + i / 5, [0.0, -0.5], 0.2, kernel_type='R', sigma=0.5)
        svm1.plot()


def Q_3():
    for i in range(1,5):
        svm1 = SVMClassifier([1.5, 0.5], 0.2, [-1.5, 0.5], 0.2, [0.0, -0.5], 0.2, kernel_type='P', p=i)
        svm1.plot()

    for i in range(1,5):
        svm1 = SVMClassifier([1.5, 0.5], 0.2, [-1.5, 0.5], 0.2, [0.0, -0.5], 0.2, kernel_type='R', sigma=i/5)
        svm1.plot()


def Q_4():
    for i in range(1,30,24):
        svm1 = SVMClassifier([1.5, 0.5], 0.2, [-1.5, 0.5], 0.2, [0.0, 0.0], 0.2, kernel_type='L', C=i)
        svm1.plot()




def main():
    Q_4()

if __name__=='__main__':
    main()