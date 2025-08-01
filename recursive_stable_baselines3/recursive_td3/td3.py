from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F


from recursive_stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from recursive_stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from recursive_stable_baselines3.recursive_td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy


from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class TD3(OffPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],

        init,
        update,
        post,

        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1e6,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        recursive_type: Union[th.device, str] = "dsum",
        output_number: int = 1,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            recursive_type=recursive_type,
            output_number=output_number,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.init = init
        self.update = update
        self.post = post

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        # print("output num", self.output_number)

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next tau-values: min over all critics targets
                tau_1_next, tau_2_next = self.critic_target(replay_data.next_observations, next_actions)  # (batch_size, 3)

                initvalue = self.init.to(tau_1_next.device)
                if initvalue.ndim == 1:  # shape (dim,)
                    initvalue = initvalue.unsqueeze(0).expand_as(tau_1_next)  # shape (batch_size, dim)

                tau_1_target = th.where(replay_data.dones.bool(), initvalue, tau_1_next)
                tau_2_target = th.where(replay_data.dones.bool(), initvalue, tau_2_next)

                tau_1_next_update = self.update(replay_data.rewards, tau_1_target)
                tau_2_next_update = self.update(replay_data.rewards, tau_2_target)

                tau_1_next_update_post = self.post(tau_1_next_update)
                tau_2_next_update_post = self.post(tau_2_next_update)

                target_tau_values = th.min(tau_1_next_update_post, tau_2_next_update_post)

            # Get current Q-values estimates for each critic network
            current_tau_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(self.post(current_tau), target_tau_values) for current_tau in current_tau_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                tau_for_actor = self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations))
                actor_loss = -self.post(tau_for_actor).mean()

                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    @classmethod
    def load_with_aggregation(cls, path, init, update, post, **kwargs):
        data, params, _ = cls.load(path, return_data=True)
        model = cls(policy=data["policy"], env=None,
                    init=init, update=update, post=post,
                    **data["init_kwargs"])
        model.set_parameters(params)
        return model
