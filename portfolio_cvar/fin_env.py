import scipy
import numpy as np
import gymnasium as gym

# This environment (FinEnv_resursive) is used in our RRA paper for portfolio optimization experiments.
# The reward at each step is the simple portfolio return, defined as the percentage change in total portfolio value.
# Specifically, R_t = (P_{t+1} - P_t) / P_t, where P_t is the portfolio value at time t.
class FinEnv_resursive(gym.Env):
    def __init__(self, start_date, end_date, data_folder,
                 adapt_state, adapt_reward,
                 lookback_window=59,
                 eta=1 / 252,
                 init_P=1e5,
                 eval=False,
                 init_past=False,
                 ):
        # Simulation and configuration parameters
        self.start_date = start_date                # Simulation start date
        self.end_date = end_date                    # Simulation end date
        self.data_folder = data_folder              # Folder containing input data files
        self.adapt_state = adapt_state              # Whether to include additional state features
        self.adapt_reward = adapt_reward            # Whether to adapt the reward
        self.lookback_window = lookback_window      # Number of past days used to form state
        self.eta = eta                              # Daily discount factor
        self.init_P = init_P                        # Initial portfolio value
        self.eval = eval                            # Evaluation mode flag
        self.init_past = init_past                  # Whether to initialize with lookback statistics

        # Load date list
        with open(self.data_folder + 'date_list.txt', 'r') as f:
            self.date_list = f.read().splitlines()

        # Load input market data
        with open(self.data_folder + 'indices_log_returns.npy', 'rb') as f:
            self.indices_log_returns = np.load(f)

        with open(self.data_folder + 'indices_share_prices.npy', 'rb') as f:
            self.indices_share_prices = np.load(f)

        with open(self.data_folder + 'vix_normed_lookback.npy', 'rb') as f:
            self.vix = np.load(f)

        with open(self.data_folder + 'vol_20_60_quotient_normed_lookback.npy', 'rb') as f:
            self.vol_20_60_quotient = np.load(f)

        with open(self.data_folder + 'vol_20_normed_lookback.npy', 'rb') as f:
            self.vol_20 = np.load(f)

        with open(self.data_folder + 'indices_simple_returns.npy', 'rb') as f:
            self.simple_returns = np.load(f)

        self.start_index = self.date_list.index(self.start_date)
        assert self.start_index >= self.lookback_window, "Start date too early"

        self.n_stocks = self.indices_log_returns.shape[1]

        self.end_index = self.date_list.index(self.end_date)

        self.current_index = self.start_index

        self.allocations = np.zeros(self.n_stocks + 1, dtype=np.float32)
        # All assets are cash
        self.allocations[-1] = 1.0

        bounds_low_log_returns = np.full(shape=self.lookback_window * self.n_stocks,
                                         fill_value=-np.inf)  # fill_value = np.min(self.indices_log_returns.shape))
        bounds_high_log_returns = np.full(shape=self.lookback_window * self.n_stocks,
                                          fill_value=np.inf)  # fill_value = np.max(self.indices_log_returns.shape))

        bounds_low_allocations = np.full(shape=self.n_stocks + 1, fill_value=0.)
        bounds_high_allocations = np.full(shape=self.n_stocks + 1, fill_value=1.)

        bounds_low_vol = np.full(shape=2, fill_value=-np.inf)
        bounds_high_vol = np.full(shape=2, fill_value=np.inf)

        bounds_low_vix = np.full(shape=self.lookback_window - 2, fill_value=-np.inf)
        bounds_high_vix = np.full(shape=self.lookback_window - 2, fill_value=np.inf)

        bounds_low_ABt = np.full(shape=3, fill_value=-np.inf)
        bounds_high_ABt = np.full(shape=3, fill_value=np.inf)

        if self.adapt_state:
            bounds_low_full = np.concatenate([bounds_low_log_returns, bounds_low_allocations,
                                              bounds_low_vol, bounds_low_vix, bounds_low_ABt])
            bounds_high_full = np.concatenate([bounds_high_log_returns, bounds_high_allocations,
                                               bounds_high_vol, bounds_high_vix, bounds_high_ABt])
            self.observation_space = gym.spaces.Box(low=bounds_low_full, high=bounds_high_full,
                                                    shape=bounds_high_full.shape)
        else:
            bounds_low_full = np.concatenate([bounds_low_log_returns, bounds_low_allocations,
                                              bounds_low_vol, bounds_low_vix])
            bounds_high_full = np.concatenate([bounds_high_log_returns, bounds_high_allocations,
                                               bounds_high_vol, bounds_high_vix])
            self.observation_space = gym.spaces.Box(low=bounds_low_full, high=bounds_high_full,
                                                    shape=bounds_high_full.shape)

        self.action_space = gym.spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max,
                                           shape=(self.n_stocks + 1,))

    def get_A_std(self):
        return (np.nan_to_num(np.mean(self.past_R), nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(np.std(self.past_R), nan=0.0, posinf=0.0, neginf=0.0))

    def _get_obs(self):
        obs = np.concatenate(
            [self.indices_log_returns[self.current_index + 1 - self.lookback_window: self.current_index + 1].flatten(),
             self.allocations,
             [self.vol_20[self.current_index],
              self.vol_20_60_quotient[self.current_index]],
             self.vix[self.current_index + 3 - self.lookback_window: self.current_index + 1],
             ])
        if self.adapt_state:
            A, std = self.get_A_std()
            obs = np.concatenate([obs, [A, std, 1 - self.t / (self.end_index - self.start_index)]])
        return obs

    def _get_info(self):
        return {"t": self.t, "R": self.R, "p": self.P}

    @property
    def t(self):
        return self.current_index - self.start_index + 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset tracking variables
        self.R = 0.0  # Last return
        self.past_R = []  # History of observed returns

        self.P = self.init_P

        self.current_index = self.start_index - 1

        return self._get_obs(), self._get_info()

    def step(self, action):

        self.current_index += 1

        # Norm actions
        actions_norm = scipy.special.softmax(action)

        # rebalance to by full shares
        money_per_share = self.P * actions_norm

        shares = money_per_share[:-1] // self.indices_share_prices[self.current_index - 1]

        P_per_share = shares * self.indices_share_prices[self.current_index - 1]
        cash = self.P - np.sum(P_per_share)
        allocations_unnormed = np.concatenate([P_per_share, [cash, ]])

        self.allocations = allocations_unnormed / np.sum(allocations_unnormed)

        # Compute new portfolio value at current timestep using updated prices
        new_p = np.sum(shares * self.indices_share_prices[self.current_index]) + cash

        # Compute simple return
        self.R = (new_p - self.P) / self.P

        # Reward is the simple portfolio return this step
        reward = self.R

        terminated = False
        if self.current_index >= self.end_index:
            terminated = True

        # Update environment state
        self.P = new_p  # Update portfolio value
        self.past_R.append(self.R)  # Store return for history

        # Generate next observation and info
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info

    def close(self):
        pass


