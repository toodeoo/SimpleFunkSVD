import numpy as np


def sgd(w, dw, config=None):
    """
    @:param w -- the weight matrix
    @:param dw -- the gradient of this matrix
    @:param config
    config format:
    - learning_rate: Scalar learning rate.
    @:returns next_w, config: the weight matrix after gradient descent
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    next_w = w - config['learning_rate'] * dw
    return next_w, config


def adam(w, dw, config=None):
    """
    @:param w -- the weight matrix
    @:param dw -- the gradient of this matrix
    @:param config
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    @:returns next_w, config: the weight matrix after gradient descent
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw ** 2
    next_w = w - config['learning_rate'] * config['m'] / (np.sqrt(config['v']) + config['epsilon'])
    return next_w, config
