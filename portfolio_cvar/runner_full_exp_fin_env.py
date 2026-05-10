import argparse
import random
import sys

import os
import copy
import pickle
import torch as th
import numpy as np
import empyrical as ep

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, REPO_ROOT)

from recursive_stable_baselines3 import Recursive_PPO_multi_output
from recursive_stable_baselines3.recursive_common.statistics_portfolio import (
    init_cvar, post_cvar, update_cvar,
    init_sharpe, post_sharpe, update_sharpe,
    init_mean_return, post_mean_return, update_mean_return,
)


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from own_eval_callback import OwnEvalCallback, own_eval_policy

from fin_env import FinEnv_resursive
from fin_utils import find_closest_date_before, find_closest_date_after, str_to_bool


init = None  # 後で初期化
update = None  # 後で初期化
post = None  # 後で初期化
output_feature_num = None  # 後で初期化
cvar_num_bins = 201  # グローバル変数


def _build_train_objective_registry(cvar_num_bins: int) -> dict:
    """学習目的のレジストリを構築する。新しい目的を追加する場合はここに追記する。

    各エントリは (init, update, post, output_feature_num) のタプル。
    """

    def _make_cvar_update(num_bins):
        def _update(rewards, tau):
            return update_cvar(rewards, tau, num_bins=num_bins)
        return _update

    def _make_cvar_post(num_bins):
        def _post(tau):
            return post_cvar(tau, num_bins=num_bins)
        return _post

    return {
        "cvar": (
            init_cvar(num_bins=cvar_num_bins, as_torch=True),
            _make_cvar_update(cvar_num_bins),
            _make_cvar_post(cvar_num_bins),
            2 * cvar_num_bins,
        ),
        "sharpe": (
            init_sharpe(),
            update_sharpe,
            post_sharpe,
            3,
        ),
        "mean_return": (
            init_mean_return(),
            update_mean_return,
            post_mean_return,
            2,
        ),
        # 新しい目的をここに追加:
        # "example": (
        #     init_example(),
        #     update_example,
        #     post_example,
        #     <output_feature_num>,
        # ),
    }


def compute_cvar_from_returns(returns):
    """Compute CVaR from a sequence of returns using NumPy-based tau."""
    global cvar_num_bins
    tau = init_cvar(num_bins=cvar_num_bins, as_torch=False)  # Use NumPy version for evaluation
    for reward in np.asarray(returns, dtype=np.float32).reshape(-1):
        tau = update_cvar(reward, tau, num_bins=cvar_num_bins)  # update_cvar handles both NumPy and Torch
    # post_cvar will return a NumPy array or scalar when input is NumPy
    cvar_value = post_cvar(tau, num_bins=cvar_num_bins)
    if isinstance(cvar_value, np.ndarray):
        cvar_value = float(cvar_value[0]) if cvar_value.size == 1 else float(cvar_value)
    else:
        cvar_value = float(cvar_value)
    return np.nan_to_num(cvar_value, nan=0.0, posinf=0.0, neginf=0.0)


def compute_sharpe_from_returns(returns):
    """Compute annualized Sharpe ratio from a sequence of daily step returns."""
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    std = np.std(returns)
    if std < 1e-10:
        return 0.0
    return float(np.nan_to_num(np.mean(returns) / std * np.sqrt(252), nan=0.0, posinf=0.0, neginf=0.0))


def compute_mean_return_from_returns(returns):
    """Compute mean step return from a sequence of step returns."""
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    return float(np.nan_to_num(np.mean(returns), nan=0.0, posinf=0.0, neginf=0.0))



def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)

def main():

    parser = argparse.ArgumentParser(description="Runner Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed_start', type=int, default=5, help='Random seed')
    parser.add_argument('--adapt_state', type=str, default="False", help='adapt state')
    parser.add_argument('--adapt_reward', type=str, default="False", help='adapt reward')
    parser.add_argument('--result_dir', type=str, default="./full_exp", help='output directory')
    parser.add_argument('--cvar_num_bins', type=int, default=201, help='Number of bins for CVaR calculation')
    parser.add_argument('--train_objective', type=str, default='cvar',
                        help='学習目的。利用可能な値: cvar, sharpe (将来追加予定: mean_return など)')


    args = vars(parser.parse_args())
    # We used seed_start [0, 5, 10, 15, 20] for the experiments
    seed_start=args["seed_start"]
    n_seeds = 1
    seeds = np.arange(seed_start, seed_start+n_seeds, 1, dtype=int)
    adapt_state=str_to_bool(args["adapt_state"])
    adapt_reward=str_to_bool(args["adapt_reward"])
    
    global init, update, post, output_feature_num, cvar_num_bins
    cvar_num_bins = args["cvar_num_bins"]
    train_objective = args["train_objective"]

    registry = _build_train_objective_registry(cvar_num_bins)
    if train_objective not in registry:
        raise ValueError(
            f"未知の train_objective: '{train_objective}'。"
            f"利用可能な値: {list(registry.keys())}"
        )
    init, update, post, output_feature_num = registry[train_objective]

    # Select per-episode metric function for eval callback best-model selection
    _metric_fn_map = {
        "cvar": compute_cvar_from_returns,
        "sharpe": compute_sharpe_from_returns,
        "mean_return": compute_mean_return_from_returns,
    }
    metric_fn_for_callback = _metric_fn_map.get(train_objective, None)

    bins_suffix = f"_bins{cvar_num_bins}" if train_objective == "cvar" else ""

    if train_objective == "cvar":
        print(f"{seed_start=}, {adapt_state=}, {adapt_reward=}, {cvar_num_bins=}, {train_objective=}", flush=True)
    else:
        print(f"{seed_start=}, {adapt_state=}, {adapt_reward=}, {train_objective=}", flush=True)


    # 5years
    start_date_list_train = ["2006-01-01", "2007-01-01", "2008-01-01", "2009-01-01", "2010-01-01", 
                    "2011-01-01", "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01",]
    end_date_list_train = ["2010-12-31", "2011-12-31", "2012-12-31", "2013-12-31", "2014-12-31",
                    "2015-12-31", "2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31",]

    # 1year
    start_date_list_eval = ["2011-01-01", "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01",
                            "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01",]
    end_date_list_eval = ["2011-12-31", "2012-12-31", "2013-12-31", "2014-12-31", "2015-12-31",
                            "2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31"]

    # 1year
    start_date_list_test = ["2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01",
                            "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01",]
    end_date_list_test = ["2012-12-31", "2013-12-31", "2014-12-31", "2015-12-31", "2016-12-31",
                        "2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]


    exp_dir = os.path.abspath(args["result_dir"])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_dir=os.path.join(exp_dir, f'log_{adapt_reward=}_{adapt_state=}_{train_objective}{bins_suffix}')
    tensorboard_dir =os.path.join(exp_dir, f"tensorboard_{adapt_reward=}_{adapt_state=}_{train_objective}{bins_suffix}")

    check_and_make_directories([log_dir, tensorboard_dir])


    verbose = 1

    n_evals = 200

    # PPO paras
    training_steps = 7.5e6
    n_envs = 10
    n_steps =  252*3
    batch_size = 252*5
    n_epochs = 16
    gamma=0.9
    gae_lambda=0.9
    clip_range=0.25

    lr_start = 3e-4
    lr_end = 1e-5

    eval_freq = int(training_steps / n_evals / n_envs)

    deterministic_eval = True
    if deterministic_eval:
        n_traj_eval = 1
    else:
        n_traj_eval = n_envs


    # shape (n_steps)
    best_model_seed = []

    # shapes (n_steps, n_seeds)
    cvar_list_train = []
    cvar_list_eval = []
    cvar_list_test = []

    # shapes (n_steps, n_seeds, n_steps)
    all_R_train = []
    all_R_eval = []
    all_R_test = []

    start_step = 0
    last_best_model_path = None

    data_folder = os.path.join(BASE_DIR, "preproc_data") + os.sep
    with open(data_folder + "date_list.txt", "r") as f:
        date_list = f.read().splitlines()

    #ステップは1step = 5year train,1year eval,1year testを1年ずつスライドさせた10step
    for step in range(start_step, len(start_date_list_train)):
        best_model_save_path = os.path.join(log_dir, f"best_model_{step}")
        tensorboard_dir_log = os.path.join(tensorboard_dir, f"tensorboard_{step}")
        check_and_make_directories([best_model_save_path, tensorboard_dir_log])

        cvar_list_train_step = []
        cvar_list_eval_step = []
        cvar_list_test_step = []

        all_R_train_step = []
        all_R_eval_step = []
        all_R_test_step = []

        # seed値ごとに環境+モデルのセットを生成
        for seed in seeds:
            seed = int(seed)
            th.manual_seed(seed)
            th.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            def lr_schedule(progress):
                return lr_end + (lr_start - lr_end) * progress

            policy_kwargs = dict(
                        net_arch=dict(pi=[64, 64], vf=[64, 64]),
                        activation_fn=th.nn.Tanh,
                        log_std_init=-1.,
            )

            start_date = find_closest_date_after(start_date_list_train[step], date_list)
            end_date = find_closest_date_before(end_date_list_train[step], date_list)

            env_kwargs = {  "start_date": start_date, 
                            "end_date": end_date, 
                            "data_folder": data_folder, 
                            "adapt_state": adapt_state, 
                            "adapt_reward": adapt_reward,
                            "lookback_window":59,
                            "eta": 1/252,
                            "init_P": 1e5,
                            "init_past": False,
                        }
            
            to_add=f"_{seed=}, {start_date=}, {end_date=}, {step=}"

            # same (training)
            # Training-time checkpoint selection follows mean reward, like portfolio_sharpe.
            eval_env_kwargs_0 = copy.deepcopy(env_kwargs)
            eval_env_kwargs_0["start_date"] = find_closest_date_after(start_date_list_train[step], date_list) 
            eval_env_kwargs_0["end_date"] = find_closest_date_before(end_date_list_train[step], date_list)
            eval_env_kwargs_0["eval"] = True
            eval_env_0 = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=eval_env_kwargs_0)
            eval_env_callback_0 = eval_env_kwargs_0
            eval_callback_0 = OwnEvalCallback(eval_env_0, best_model_save_path=best_model_save_path,
                                        log_path=best_model_save_path, eval_freq=eval_freq, n_eval_episodes=n_traj_eval,
                                        deterministic=deterministic_eval, render=False, to_add= "_same" + to_add, verbose=verbose,
                                        tens_name=f"same_{step}",
                                        metric=train_objective,
                                        single_eval_env_fn=lambda kwargs=eval_env_callback_0: FinEnv_resursive(**kwargs),
                                        metric_fn=metric_fn_for_callback,)

            # eval (model selection)
            eval_env_kwargs_1 = copy.deepcopy(env_kwargs)
            eval_env_kwargs_1["start_date"] = find_closest_date_after(start_date_list_eval[step], date_list) 
            eval_env_kwargs_1["end_date"] = find_closest_date_before(end_date_list_eval[step], date_list)
            eval_env_kwargs_1["eval"] = True
            eval_env_1 = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=eval_env_kwargs_1)

            eval_env_callback_1 = eval_env_kwargs_1
            eval_callback_1 = OwnEvalCallback(eval_env_1, best_model_save_path=best_model_save_path,
                                        log_path=best_model_save_path, eval_freq=eval_freq, n_eval_episodes=n_traj_eval,
                                        deterministic=deterministic_eval, render=False, to_add="_eval" + to_add, verbose=verbose,
                                        tens_name=f"eval_{step}",
                                        metric=train_objective,
                                        single_eval_env_fn=lambda kwargs=eval_env_callback_1: FinEnv_resursive(**kwargs),
                                        metric_fn=metric_fn_for_callback,)

            # test (the test split is logged for analysis, not for checkpoint selection)
            eval_env_kwargs_2 = copy.deepcopy(env_kwargs)
            eval_env_kwargs_2["start_date"] = find_closest_date_after(start_date_list_test[step], date_list) 
            eval_env_kwargs_2["end_date"] = find_closest_date_before(end_date_list_test[step], date_list)
            eval_env_kwargs_2["eval"] = True
            eval_env_2 = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv,
                                    env_kwargs=eval_env_kwargs_2)
            eval_env_callback_2 = eval_env_kwargs_2
            eval_callback_2 = OwnEvalCallback(eval_env_2, best_model_save_path=best_model_save_path,
                                        log_path=best_model_save_path, eval_freq=eval_freq, n_eval_episodes=n_traj_eval,
                                        deterministic=deterministic_eval, render=False, to_add= "_test" + to_add, verbose=verbose,
                                        tens_name=f"test_{step}",
                                        metric=train_objective,
                                        single_eval_env_fn=lambda kwargs=eval_env_callback_2: FinEnv_resursive(**kwargs),
                                        metric_fn=metric_fn_for_callback,)

            vec_env = make_vec_env(FinEnv_resursive, n_envs=n_envs, monitor_dir=log_dir, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

            model = Recursive_PPO_multi_output("MlpPolicy_multi_output",
                        vec_env,

                        init = init,
                        update = update,
                        post = post,

                        tensorboard_log = tensorboard_dir_log,
                        learning_rate = copy.deepcopy(lr_schedule),
                        n_steps = n_steps,
                        batch_size = batch_size,
                        n_epochs = n_epochs,
                        gamma = gamma,
                        gae_lambda = gae_lambda,
                        clip_range = clip_range,
                        policy_kwargs = policy_kwargs,
                        seed = seed,
                        output_feature_num=output_feature_num,
                                               )
            
            if last_best_model_path:
                model.set_parameters(last_best_model_path)
            model.set_random_seed(seed)

            # モデルの学習
            model.learn(total_timesteps=training_steps,
                        callback=[eval_callback_0, eval_callback_1, eval_callback_2,])

            eval_env_0.close()
            eval_env_1.close()
            eval_env_2.close()
            vec_env.close()

            # Post-training CVaR analysis for the best checkpoints saved during training.
            model.set_parameters(os.path.join(best_model_save_path, "best_model" + "_same" + to_add))
            eval_env_0 = FinEnv_resursive(**eval_env_kwargs_0)
            _, _, R_same = own_eval_policy(
                model,
                eval_env_0,
                deterministic=deterministic_eval,
            )
            cvar_same = compute_cvar_from_returns(R_same)

            model.set_parameters(os.path.join(best_model_save_path, "best_model" + "_eval" + to_add))
            eval_env_1 = FinEnv_resursive(**eval_env_kwargs_1)
            _, _, R_eval = own_eval_policy(
                model,
                eval_env_1,
                deterministic=deterministic_eval,
            )
            cvar_eval = compute_cvar_from_returns(R_eval)

            model.set_parameters(os.path.join(best_model_save_path, "best_model" + "_eval" + to_add))
            eval_env_2 = FinEnv_resursive(**eval_env_kwargs_2)
            _, _, R_test = own_eval_policy(
                model,
                eval_env_2,
                deterministic=deterministic_eval,
            )
            cvar_test = compute_cvar_from_returns(R_test)


            cvar_list_train_step.append(cvar_same)
            cvar_list_eval_step.append(cvar_eval)
            cvar_list_test_step.append(cvar_test)

            all_R_train_step.append(R_same)
            all_R_eval_step.append(R_eval)
            all_R_test_step.append(R_test)

            
        cvar_list_train.append(cvar_list_train_step)
        cvar_list_test.append(cvar_list_test_step)
        cvar_list_eval.append(cvar_list_eval_step)

        all_R_train.append(all_R_train_step)
        all_R_eval.append(all_R_eval_step)
        all_R_test.append(all_R_test_step)

        # Choose the next seed to carry forward using the eval CVaR.
        best_seed = seeds[np.argmax(cvar_list_eval_step)]
        best_model_seed.append(best_seed)
        to_add_best=f"_seed={best_seed}, {start_date=}, {end_date=}, {step=}"
        last_best_model_path = os.path.join(best_model_save_path, "best_model" + "_eval" + to_add_best)  

        # Save CVaR scores
        cvar_train_np = np.array(cvar_list_train)
        cvar_test_np = np.array(cvar_list_test)
        cvar_eval_np = np.array(cvar_list_eval)

        best_model_seed_np = np.array(best_model_seed)

        with open(os.path.join(log_dir, "cvar_train.npy"), "wb") as f:
            np.save(f, cvar_train_np)
        with open(os.path.join(log_dir, "cvar_test.npy"), "wb") as f:
            np.save(f, cvar_test_np)
        with open(os.path.join(log_dir, "cvar_eval.npy"), "wb") as f:
            np.save(f, cvar_eval_np)

        with open(os.path.join(log_dir, "all_R_train.pkl"), "wb") as f:
            pickle.dump(all_R_train, f)
        with open(os.path.join(log_dir, "all_R_test.pkl"), "wb") as f:
            pickle.dump(all_R_test, f)
        with open(os.path.join(log_dir, "all_R_eval.pkl"), "wb") as f:
            pickle.dump(all_R_eval, f)

        with open(os.path.join(log_dir, "best_model_seed.npy"), "wb") as f:
            np.save(f, best_model_seed_np)

if __name__ == '__main__':
    main()

