import operator
from collections import Mapping, Iterable
from functools import partial, reduce
from itertools import product
from sklearn.utils import check_random_state


class ParameterGrid:
    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))
        self.param_grid = param_grid

    def __iter__(self):
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)


class ParameterSampler:
    def __init__(self, param_distributions, n_iter, *, random_state=None):
        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError('Parameter distribution is not a dict or '
                            'a list ({!r})'.format(param_distributions))

        if isinstance(param_distributions, Mapping):
            param_distributions = [param_distributions]

        for dist in param_distributions:
            if not isinstance(dist, dict):
                raise TypeError('Parameter distribution is not a '
                                'dict ({!r})'.format(dist))
            for key in dist:
                if (not isinstance(dist[key], Iterable)
                        and not hasattr(dist[key], 'rvs')):
                    raise TypeError('Parameter value is not iterable '
                                    'or distribution (key={!r}, value={!r})'
                                    .format(key, dist[key]))
        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    def _is_all_lists(self):
        return all(
            all(not hasattr(v, "rvs") for v in dist.values())
            for dist in self.param_distributions
        )

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_iter):
            dist = rng.choice(self.param_distributions)
            items = sorted(dist.items())
            params = dict()
            for k, v in items:
                if hasattr(v, "rvs"):
                    params[k] = v.rvs(random_state=rng)
                else:
                    params[k] = v[rng.randint(len(v))]
            yield params

    def __len__(self):
        if self._is_all_lists():
            grid_size = len(ParameterGrid(self.param_distributions))
            return min(self.n_iter, grid_size)
        else:
            return self.n_iter
