import numpy as np
import optimal
from sklearn.metrics import mean_squared_error


class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.X_val = data['X_val']

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optimal_config = kwargs.pop('optimal_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_val_samples = kwargs.pop('num_val_samples', 1000)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        self.update_rule = getattr(optimal, self.update_rule)

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_rmse = 1e5
        self.best_params = {}
        self.loss_history = []
        self.val_rmse_history = []

        self.optimal_config = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optimal_config.items()}
            self.optimal_config[p] = d

    def _step(self):
        num_row, num_col = self.X_train.shape[0], self.X_train.shape[1]
        batch_mask_row = np.random.choice(num_row, self.batch_size)
        batch_mask_col = np.random.choice(num_col, self.batch_size)
        X_batch = np.stack((np.array(batch_mask_row), np.array(batch_mask_col)), axis=1).reshape(self.batch_size, 2)
        loss, grads = self.model.loss(X_batch)
        if loss > 0:
            self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optimal_config[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optimal_config[p] = next_config

    def _val_checkpoint(self, num_samples):
        N, M = self.X_val.shape[0], self.X_val.shape[1]
        if num_samples is not None and N > num_samples and M > num_samples:
            batch_mask_row = np.random.choice(N, num_samples)
            batch_mask_col = np.random.choice(M, num_samples)
            X_batch = np.stack((np.array(batch_mask_row), np.array(batch_mask_col)), axis=1).reshape(num_samples, 2)
        else:
            raise ValueError("num_samples is not None")
        y_pred = []
        y_real = []
        for _ in range(X_batch.shape[0]):
            y_pred.append(self.model.predict(X_batch[_, 0], X_batch[_, 1]))
            y_real.append(self.X_val[X_batch[_, 0], X_batch[_, 1]])

        return np.sqrt(mean_squared_error(np.array(y_real), np.array(y_pred)))

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optimal_config:
                    self.optimal_config[k]['learning_rate'] *= self.lr_decay

            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                val_rmse = self._val_checkpoint(num_samples=self.num_val_samples)
                self.val_rmse_history.append(val_rmse)
                # self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) val_rse: %f' % (
                        self.epoch, self.num_epochs, val_rmse))

                # Keep track of the best model
                if val_rmse < self.best_val_rmse:
                    self.best_val_rmse = val_rmse
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
        self.model.params = self.best_params
