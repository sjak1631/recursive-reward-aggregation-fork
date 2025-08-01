import torch as th

def softplus(x):
    return th.nn.functional.softplus(th.tensor(x, dtype=th.float32))


# ----------------------- init ----------------------- #

def init_dsum():
    init_tau = th.tensor(0.0)
    return init_tau

# dmax, log_sum_exp
def init_dmax():
    init_tau = th.tensor(float('-inf'))
    return init_tau

def init_min():
    init_tau = th.tensor(float('inf'))
    return init_tau

def init_dsum_dmax():
    init_tau_sum = th.tensor(0.0)
    init_tau_max = th.tensor(float('-inf'))
    init_tau = th.stack([init_tau_sum, init_tau_max])
    return init_tau

# min_max, range
def init_min_max():
    init_tau_min = th.tensor(float('inf'))
    init_tau_max = th.tensor(float('-inf'))
    init_tau = th.stack([init_tau_min, init_tau_max])
    return init_tau

def init_mean():
    init_tau_sum = th.tensor(0.0)
    init_tau_length = th.tensor(0.0)
    init_tau = th.stack([init_tau_sum, init_tau_length])
    return init_tau

def init_dsum_variance():
    init_tau_sum = th.tensor(0.0)
    init_tau_mean = th.tensor(0.0)
    init_tau_variance = th.tensor(0.0)
    init_tau_length = th.tensor(0.0)
    init_tau = th.stack([init_tau_sum, init_tau_mean, init_tau_variance, init_tau_length])
    return init_tau

def init_sharpe():
    init_tau_mean = th.tensor(0.0)
    init_tau_variance = th.tensor(0.0)
    init_tau_length = th.tensor(0.0)
    init_tau = th.stack([init_tau_mean, init_tau_variance, init_tau_length])
    return init_tau


# ----------------------- update ----------------------- #

def update_dsum(gamma: float):
    def update(rewards, tau):
        update_tau = tau.clone()
        update_tau = rewards + gamma * update_tau
        return update_tau
    return update

def update_dmax(gamma: float):
    def update(rewards, tau):
        update_tau = tau.clone()
        update_tau = th.maximum(rewards, gamma * update_tau)
        return update_tau
    return update

def update_min(rewards, tau):
    update_tau = tau.clone()
    update_tau = th.minimum(rewards, update_tau)
    return update_tau

def update_log_sum_exp(rewards, tau):
    update_tau = tau.clone()
    update_tau = th.logsumexp(th.stack([rewards, update_tau], dim=0), dim=0)
    return update_tau

def update_dsum_dmax(gamma: float):
    def update(rewards, tau):
        update_tau = tau.clone()
        tau_dsum, tau_dmax = tau[:, 0], tau[:, 1]
        rewards = rewards.squeeze()
        update_tau[:, 0] = rewards + gamma * tau_dsum
        update_tau[:, 1] = th.maximum(rewards, gamma * tau_dmax)
        return update_tau
    return update

# min_max, range
def update_min_max(rewards, tau):
    update_tau = tau.clone()
    tau_min, tau_max = tau[:, 0], tau[:, 1]
    rewards = rewards.squeeze()
    update_tau[:, 0] = th.min(rewards, tau_max)
    update_tau[:, 1] = th.maximum(rewards, tau_max)
    return update_tau

def update_mean(rewards, tau):
    update_tau = tau.clone()
    tau_sum, tau_length = tau[:, 0], tau[:, 1]
    tau_length = softplus(tau_length)
    rewards = rewards.squeeze()
    update_tau[:, 0] = rewards + tau_sum
    update_tau[:, 1] = 1 + tau_length
    return update_tau

def update_dsum_variance(gamma: float):
    def update(rewards, tau):
        update_tau = tau.clone()
        tau_dsum, tau_mean, tau_variance, tau_length = tau[:, 0], tau[:, 1], tau[:, 2], tau[:, 3]
        tau_length = softplus(tau_length)
        rewards = rewards.squeeze()
        update_tau[:, 3] = 1 + tau_length
        update_tau[:, 0] = rewards + gamma * tau_dsum
        update_tau[:, 1] = tau_mean + ((rewards - tau_mean) / update_tau[:, 3])
        update_tau[:, 2] = tau_variance + (
                (rewards - update_tau[:, 1]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 3]
        return update_tau
    return update

def update_sharpe(rewards, tau):
    update_tau = tau.clone()
    tau_mean, tau_variance, tau_length = tau[:, 0], tau[:, 1], tau[:, 2]
    tau_length = softplus(tau_length)
    update_tau[:, 2] = 1 + tau_length
    update_tau[:, 0] = tau_mean + ((rewards - tau_mean) / update_tau[:, 2])
    update_tau[:, 1] = tau_variance + ((rewards - update_tau[:, 0]) * (rewards - tau_mean) - tau_variance) / update_tau[:, 2]
    return update_tau


# ----------------------- post ----------------------- #

# dsum, dmax, min, log_sum_exp
def post_id(tau):
    return tau

def post_dsum_dmax(lam):
    def post(tau):
        tau_dsum, tau_dmax = tau[..., 0], tau[..., 1]
        post_tau = tau_dsum + ((1 - lam)/lam) * tau_dmax
        return post_tau
    return post

def post_min_max(lam):
    def post(tau):
        tau_min, tau_max = tau[..., 0], tau[..., 1]
        post_tau = lam * tau_min + (1 - lam) * tau_max
        return post_tau
    return post

def post_mean(tau):
    tau_sum, tau_length = tau[..., 0], tau[..., 1]
    post_tau = tau_sum / tau_length
    return post_tau

def post_range(tau):
    tau_min, tau_max = tau[..., 0], tau[..., 1]
    post_tau = tau_max - tau_min
    return post_tau

def post_dsum_variance(tau):
    tau_dsum, tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2], tau[..., 3]
    tau_variance = softplus(tau_variance)
    post_tau = tau_dsum - tau_variance
    return post_tau

def post_sharpe(tau):
    tau_mean, tau_variance, tau_length = tau[..., 0], tau[..., 1], tau[..., 2]
    tau_variance = softplus(tau_variance)
    post_tau = tau_mean / th.sqrt(tau_variance + 1e-8)
    return post_tau
