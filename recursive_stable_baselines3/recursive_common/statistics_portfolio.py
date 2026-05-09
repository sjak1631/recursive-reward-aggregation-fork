import torch as th
import numpy as np

def softplus(x):
    if not isinstance(x, th.Tensor):
        x = th.tensor(x, dtype=th.float32)
    return th.nn.functional.softplus(x.float())


# Bin-based CVaR approximation for episodic reward distributions.
CVAR_ALPHA = 0.05
CVAR_REWARD_MIN = -1.0
CVAR_REWARD_MAX = 1.0


# 入力をnumpyにそろえる
def _as_numpy(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def _prepare_cvar_tau(tau):
    tau_array = _as_numpy(tau).astype(np.float32, copy=True)
    squeezed = tau_array.ndim == 1
    if squeezed:
        tau_array = tau_array[None, :]
    return tau_array, squeezed


def _prepare_cvar_rewards(rewards, batch_size: int):
    reward_array = _as_numpy(rewards).astype(np.float32, copy=False).reshape(-1)
    if reward_array.size == 1 and batch_size > 1:
        reward_array = np.repeat(reward_array, batch_size)
    if reward_array.size != batch_size:
        raise ValueError(f"Expected {batch_size} rewards, got {reward_array.size}.")
    return reward_array


def _cvar_bin_centers(num_bins: int, reward_min: float, reward_max: float):
    bin_width = (reward_max - reward_min) / num_bins
    return reward_min + (np.arange(num_bins, dtype=np.float32) + 0.5) * bin_width


def init_cvar(num_bins: int = 201, as_torch: bool = True):
    if as_torch: #torchの場合
        #出現回数
        init_tau_count = th.zeros(num_bins, dtype=th.float32)
        #報酬の合計
        init_tau_sum = th.zeros(num_bins, dtype=th.float32)
        return th.cat([init_tau_count, init_tau_sum]).float()
    else:#numpyの場合
        init_tau_count = np.zeros(num_bins, dtype=np.float32)
        init_tau_sum = np.zeros(num_bins, dtype=np.float32)
        return np.concatenate([init_tau_count, init_tau_sum]).astype(np.float32)

#ヒストグラムの情報を更新
def update_cvar(
    rewards,
    tau,
    num_bins: int = 201,
    reward_min: float = CVAR_REWARD_MIN,
    reward_max: float = CVAR_REWARD_MAX,
):
    #前処理
    is_torch_tensor = isinstance(tau, th.Tensor)
    device = tau.device if is_torch_tensor else None
    dtype = tau.dtype if is_torch_tensor else None
    
    update_tau, squeezed = _prepare_cvar_tau(tau)
    reward_array = _prepare_cvar_rewards(rewards, update_tau.shape[0])

    #counts/sumに分割
    counts = update_tau[:, :num_bins]
    sums = update_tau[:, num_bins: 2 * num_bins]

    #rewardsのクリップ
    clipped_rewards = np.clip(reward_array, reward_min, reward_max)
    bin_width = (reward_max - reward_min) / num_bins
    bin_indices = np.floor((clipped_rewards - reward_min) / bin_width).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    row_indices = np.arange(update_tau.shape[0])
    counts[row_indices, bin_indices] += 1.0
    sums[row_indices, bin_indices] += clipped_rewards

    if squeezed:
        result = update_tau[0].astype(np.float32)
        if is_torch_tensor:
            result = th.from_numpy(result).to(dtype=dtype, device=device)
    else:
        result = update_tau.astype(np.float32)
        if is_torch_tensor:
            result = th.from_numpy(result).to(dtype=dtype, device=device)
    
    return result

#ヒストグラムから、実際にCVaRヲケイサン
def post_cvar(
    tau,
    alpha: float = CVAR_ALPHA,
    num_bins: int = 201,
    reward_min: float = CVAR_REWARD_MIN,
    reward_max: float = CVAR_REWARD_MAX,
):
    #前処理
    is_torch_tensor = isinstance(tau, th.Tensor)
    device = tau.device if is_torch_tensor else None
    dtype = tau.dtype if is_torch_tensor else None
    
    tau_array, squeezed = _prepare_cvar_tau(tau)
    counts = tau_array[:, :num_bins]
    sums = tau_array[:, num_bins: 2 * num_bins]

    total_count = counts.sum(axis=1)
    bin_centers = _cvar_bin_centers(num_bins, reward_min, reward_max)
    bin_mean = np.where(counts > 0, sums / np.maximum(counts, 1e-8), bin_centers[None, :])

    tail_target = alpha * total_count
    previous_cumulative = np.cumsum(counts, axis=1) - counts
    tail_counts = np.clip(np.minimum(counts, tail_target[:, None] - previous_cumulative), 0.0, counts)
    tail_sum = np.sum(tail_counts * bin_mean, axis=1)
    tail_mass = np.minimum(total_count, tail_target)

    post_tau = np.divide(tail_sum, tail_mass, out=np.zeros_like(tail_sum), where=tail_mass > 0)

    if squeezed:
        result = float(post_tau[0])
        if is_torch_tensor:
            result = th.tensor(result, dtype=dtype, device=device)
    else:
        result = post_tau.astype(np.float32)
        if is_torch_tensor:
            result = th.from_numpy(result).to(dtype=dtype, device=device)
    
    return result

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


def init_mean_return():
    init_tau_mean = th.tensor(0.0)
    init_tau_length = th.tensor(0.0)
    return th.stack([init_tau_mean, init_tau_length])

def update_mean_return(rewards, tau):
    update_tau = tau.copy()
    tau_mean, tau_length = tau[:, 0], tau[:, 1]
    tau_length = softplus(tau_length)
    update_tau[:, 1] = 1 + tau_length
    update_tau[:, 0] = tau_mean + (rewards - tau_mean) / update_tau[:, 1]
    return update_tau

def post_mean_return(tau):
    return tau[..., 0]
