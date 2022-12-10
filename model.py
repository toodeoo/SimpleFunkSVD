import numpy as np


class FunkSVD(object):
    def __init__(self, R, K=20, reg_p=0, reg_q=0):
        self.R = R
        self.N = R.shape[0]
        self.M = R.shape[1]
        self.K = K
        self.reg_p = reg_p
        self.reg_q = reg_q
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
                diff = self.R[i, j] - np.dot(self.params['P'][i, :], self.params['Q'][:, j])
                loss += np.power(diff, 2) / 2 + (self.reg_p * np.power(np.linalg.norm(self.params['P'][i, :], 2), 2)
                                                 + self.reg_q * np.power(np.linalg.norm(self.params['Q'][:, j], 2),
                                                                         2)) / 2
                grad_P[i, :] = -1 * diff * self.params['Q'][:, j] + self.reg_p * self.params['P'][i, :]
                gran_Q[:, j] = -1 * diff * self.params['P'][i, :] + self.reg_q * self.params['Q'][:, j]
        grad['P'], grad['Q'] = grad_P, gran_Q
        return loss, grad

    def predict(self, i, j):
        if i >= self.params['P'].shape[0]:
            return np.means(self.params['Q'][:, j])
        elif j >= self.params['Q'].shape[1]:
            return np.means(self.params['P'][i, :])
        return np.dot(self.params['P'][i, :], self.params['Q'][:, j])


class BiasSVD(FunkSVD):
    def __init__(self, R):
        super().__init__(R)
        self.mean = R.sum() / (R != 0).sum()
        self.params['b_p'] = np.zeros(self.N)
        self.params['b_q'] = np.zeros(self.M)

    def loss(self, X_batch):
        loss, grad = 0, {}
        grad_P, grad_Q = np.zeros((self.N, self.K)), np.zeros((self.K, self.M))
        grad_bq, grad_bp = np.zeros(self.M), np.zeros(self.N)
        for _ in range(X_batch.shape[0]):
            i, j = X_batch[_][0], X_batch[_][1]
            if self.R[i, j] > 0:
                diff = self.R[i, j] - np.dot(self.params['P'][i, :], self.params['Q'][:, j]) - \
                       self.mean - self.params['b_p'][i] - self.params['b_q'][j]
                regs =\
                    self.reg_p * (np.power(np.linalg.norm(self.params['P'][i, :], 2), 2) +
                                  self.params['b_p'][i] ** 2) + \
                    self.reg_q * (np.power(np.linalg.norm(self.params['Q'][:, j], 2), 2) +
                                  self.params['b_q'][j] ** 2)

                loss += np.power(diff, 2) / 2 + regs / 2
                grad_P[i, :] = -1 * diff * self.params['Q'][:, j] + self.reg_p * self.params['P'][i, :]
                grad_Q[:, j] = -1 * diff * self.params['P'][i, :] + self.reg_q * self.params['Q'][:, j]
                grad_bp[i] = -1 * diff + self.reg_p * self.params['b_p'][i]
                grad_bp[j] = -1 * diff + self.reg_q * self.params['b_q'][j]

        grad['P'], grad['Q'] = grad_P, grad_Q
        grad['b_p'], grad['b_q'] = grad_bp, grad_bq
        return loss, grad

    def predict(self, i, j):
        if i >= self.params['P'].shape[0] or j >= self.params['Q'].shape[1]:
            return self.mean
        return np.dot(self.params['P'][i, :],
                      self.params['Q'][:, j]) + self.params['b_p'][i] + self.params['b_q'][j] + self.mean
