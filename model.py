import numpy as np


class FunkSVD(object):
    def __init__(self, R, K=20, reg=0):
        self.R = R
        self.N = R.shape[0]
        self.M = R.shape[1]
        self.K = K
        self.reg = reg
        self.params = {}
        self.params['P'] = np.random.rand(self.N, K)
        self.params['Q'] = np.random.rand(K, self.M)

    def loss(self, X_batch):
        loss, grad = 0, {}
        grad_P, gran_Q = np.zeros((self.N, self.K)), np.zeros((self.K, self.M))
        grad['P'], grad['Q'] = grad_P, gran_Q
        for _ in range(X_batch.shape[0]):
            i, j = X_batch[_][0], X_batch[_][1]
            if self.R[i, j] > 0:
                diff = np.power(self.R[i, j] - np.dot(self.params['P'][i, :], self.params['Q'][:, j]), 2) / 2
                loss += diff
                grad_P[i, :] = -1 * diff * self.params['Q'][:, j]
                gran_Q[:, j] = -1 * diff * self.params['P'][i, :]
        grad['P'], grad['Q'] = grad_P, gran_Q
        loss += self.reg * (np.linalg.norm(self.params['P'], 2) + np.linalg.norm(self.params['Q'], 2)) ** 2
        return loss, grad

    def predict(self, i, j):
        if i >= self.params['P'].shape[0]:
            return np.means(self.params['Q'][:, j])
        return np.dot(self.params['P'][i, :], self.params['Q'][:, j])
