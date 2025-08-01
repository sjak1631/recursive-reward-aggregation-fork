import torch as th
import numpy as np

def softplus(x):
    return th.nn.functional.softplus(th.tensor(x, dtype=th.float32))

def init_sharpe():
    init_tau_mean = th.tensor(0.0)
    init_tau_variance = th.tensor(0.0)
    init_tau_length = th.tensor(0.0)
    init_tau = th.stack([init_tau_mean, init_tau_variance, init_tau_length])
    return init_tau

def update_sharpe(rewards, tau):
    update_tau = tau.copy()
    tau_mean, tau_variance, tau_length = tau[:, 0], tau[:, 1], tau[:, 2]
    tau_length = softplus(tau_length)
    update_tau[:, 2] = 1 + tau_length
    update_tau[:, 0] = tau_mean + ((rewards - tau_mean) / update_tau[:, 2])
    update_tau[:, 1] = tau_variance + ((rewards - update_tau[:, 0]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 2]
    return update_tau

def post_sharpe(tau):
    tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2]
    tau_variance = softplus(tau_variance)
    post_tau = tau_mean / th.sqrt(tau_variance + 1e-8)
    return post_tau
