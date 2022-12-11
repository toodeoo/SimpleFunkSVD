import numpy as np


class FunkSVD(object):
    def __init__(self, R, K, reg_p=0, reg_q=0):
        self.R = R
        self.N = R.shape[0]
        self.M = R.shape[1]
        self.K = K
        self.reg_p = reg_p
        self.reg_q = reg_q
        self.params = {}
        self.params['P'] = np.random.rand(self.N, K)
        self.params['Q'] = np.random.rand(K, self.M)
        self.mean = R.sum() / (R != 0).sum()

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
        if i >= self.params['P'].shape[0] or j >= self.params['Q'].shape[1]:
            score = self.mean
        else:
            score = np.dot(self.params['P'][i, :], self.params['Q'][:, j])
        if score < 1:
            score = 1
        elif score > 5:
            score = 5
        return score


class BiasSVD(FunkSVD):
    def __init__(self, R, K):
        super().__init__(R, K)
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
                regs = \
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
            score = self.mean
        else:
            score = np.dot(self.params['P'][i, :],
                           self.params['Q'][:, j]) + self.params['b_p'][i] + self.params['b_q'][j] + self.mean
        if score < 1:
            score = 1
        elif score > 5:
            score = 5
        return score


class SVDpp(object):
    def __init__(self, epoch, eta, userNums, itemNums, ku=0.001, km=0.001, f=30):
        self.epoch = epoch
        self.userNums = userNums
        self.itemNums = itemNums
        self.eta = eta
        self.ku = ku
        self.km = km
        self.f = f

        self.U = None
        self.M = None

    def fit(self, train, val=None):
        self.Udict = {}
        for i in range(train.shape[0]):
            uid = train[i, 0]
            iid = train[i, 1]
            self.Udict.setdefault(uid, [])
            self.Udict[uid].append(iid)

        rateNums = train.shape[0]
        self.meanV = np.sum(train[:, 2]) / rateNums
        init_v = np.sqrt((self.meanV - 1) / self.f)
        self.U = init_v + np.random.uniform(-0.01, 0.01, (self.userNums + 1, self.f))
        self.M = init_v + np.random.uniform(-0.01, 0.01, (self.itemNums + 1, self.f))
        self.bu = np.zeros(self.userNums + 1)
        self.bi = np.zeros(self.itemNums + 1)
        self.y = np.zeros((self.itemNums + 1, self.f)) + 0.1

        for i in range(self.epoch):
            sumRmse = 0.0
            for sample in train:
                uid = sample[0]
                iid = sample[1]
                vij = float(sample[2])

                sumYj, sqrt_Ni = self.get_Yi(uid)
                p = self.meanV + self.bu[uid] + self.bi[iid] + \
                    np.sum(self.M[iid] * (self.U[uid] + sumYj))

                error = vij - p
                sumRmse += error ** 2

                deltaU = error * self.M[iid] - self.ku * self.U[uid]
                deltaM = error * (self.U[uid] + sumYj) - self.km * self.M[iid]

                self.U[uid] += self.eta * deltaU
                self.M[iid] += self.eta * deltaM

                self.bu[uid] += self.eta * (error - self.ku * self.bu[uid])
                self.bi[iid] += self.eta * (error - self.km * self.bi[iid])

                rating_list = self.Udict[uid]
                self.y[rating_list] += self.eta * (error * self.M[rating_list] / sqrt_Ni
                                                   - self.ku * self.y[rating_list])

            trainRmse = np.sqrt(sumRmse / rateNums)

            if val.any():
                _, valRmse = self.evaluate(val)
                print("Epoch %d , train RMSE: %.4f, validation RMSE: %.4f" % \
                      (i, trainRmse, valRmse))
            else:
                print("Epoch %d , train RMSE: %.4f" % \
                      (i, trainRmse))

    def evaluate(self, val):
        loss = 0
        pred = []
        for sample in val:
            uid = sample[0]
            iid = sample[1]
            if uid > self.userNums or iid > self.itemNums:
                continue
            sumYj, _ = self.get_Yi(uid)
            pred_i = self.meanV + self.bu[uid] + self.bi[iid] + np.sum(self.M[iid] * (self.U[uid] + sumYj))
            pred.append(pred_i)

            if val.shape[1] == 3:
                vij = sample[2]
                loss += (pred_i - vij) ** 2

        if val.shape[1] == 3:
            rmse = np.sqrt(loss / val.shape[0])
            return pred, rmse

        return pred

    def predict(self, test):
        return self.evaluate(test)

    def get_Yi(self, uid):
        Ni = self.Udict[uid]
        numNi = len(Ni)
        sqrt_Ni = np.sqrt(numNi)
        yj = np.zeros((1, self.f))
        if numNi == 0:
            sumYj = yj + 0.1
        else:
            yj = np.mean(self.y[Ni], axis=0)
            sumYj = yj / sqrt_Ni
        return sumYj, sqrt_Ni
