from random import random

import numpy as np
import torch
from torchrl.data import ReplayBuffer

from tensor_beasts.rl.iqn_agent import IQN


class Trainer:
    __slots__ = (
        "model",
        "model2",
        "optimizer",
        "scaler",
        "batch_size",
        "iqn_k",
        "iqn_n",
        "iqn_kappa",
        "epsilon",
        "epsilon_boltzmann",
        "gamma",
        "AL_alpha",
        "tau_epsilon_boltzmann",
        "tau_greedy_boltzmann",
        "execution_stream",
    )

    def __init__(
        self,
        model: IQN,
        model2: IQN,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
        batch_size: int,
        iqn_k: int,
        iqn_n: int,
        iqn_kappa: float,
        epsilon: float,
        epsilon_boltzmann: float,
        gamma: float,
        AL_alpha: float,
        tau_epsilon_boltzmann: float,
        tau_greedy_boltzmann: float,
    ):
        self.model = model
        self.model2 = model2
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.iqn_k = iqn_k
        self.iqn_n = iqn_n
        self.iqn_kappa = iqn_kappa
        self.epsilon = epsilon
        self.epsilon_boltzmann = epsilon_boltzmann
        self.gamma = gamma
        self.AL_alpha = AL_alpha
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.tau_greedy_boltzmann = tau_greedy_boltzmann
        self.execution_stream = torch.cuda.Stream()

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            (
                state_img_tensor,
                state_float_tensor,
                new_actions,
                new_n_steps,
                rewards_per_n_steps,
                next_state_img_tensor,
                next_state_float_tensor,
                gammas_per_n_steps,
                minirace_min_time_actions,
            ) = buffer.sample(self.batch_size)
            new_actions = new_actions.to(dtype=torch.int64)
            new_n_steps = new_n_steps.to(dtype=torch.int64)
            minirace_min_time_actions = minirace_min_time_actions.to(dtype=torch.int64)

            new_xxx = (
                torch.rand(size=minirace_min_time_actions.shape).to(device="cuda")
                * (misc.temporal_mini_race_duration_actions - minirace_min_time_actions)
            ).to(dtype=torch.int64, device="cuda")
            temporal_mini_race_current_time_actions = misc.temporal_mini_race_duration_actions - 1 - new_xxx
            temporal_mini_race_next_time_actions = temporal_mini_race_current_time_actions + new_n_steps

            state_float_tensor[:, 0] = temporal_mini_race_current_time_actions
            next_state_float_tensor[:, 0] = temporal_mini_race_next_time_actions

            new_done = temporal_mini_race_next_time_actions >= misc.temporal_mini_race_duration_actions
            possibly_reduced_n_steps = (
                new_n_steps - (temporal_mini_race_next_time_actions - misc.temporal_mini_race_duration_actions).clip(min=0)
            ).to(dtype=torch.int64)

            rewards = rewards_per_n_steps.gather(1, (possibly_reduced_n_steps - 1).unsqueeze(-1)).repeat(
                [self.iqn_n, 1]
            )  # (batch_size*iqn_n, 1)     a,b,c,d devient a,b,c,d,a,b,c,d,a,b,c,d,...
            # (batch_size*iqn_n, 1)
            gammas_pow_nsteps = gammas_per_n_steps.gather(1, (possibly_reduced_n_steps - 1).unsqueeze(-1)).repeat([self.iqn_n, 1])
            done = new_done.reshape(-1, 1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
            actions = new_actions[:, None]  # (batch_size, 1)
            actions_n = actions.repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)

            #   Use model to choose an action for next state.
            #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
            a__tpo__model__reduced_repeated = (
                self.model(
                    next_state_img_tensor,
                    next_state_float_tensor,
                    self.iqn_n,
                    tau=None,
                )[0]
                .reshape([self.iqn_n, self.batch_size, self.model.n_actions])
                .mean(dim=0)
                .argmax(dim=1, keepdim=True)
                .repeat([self.iqn_n, 1])
            )  # (iqn_n * batch_size, 1)

            #   Use model2 to evaluate the action chosen, per quantile.
            q__stpo__model2__quantiles_tau2, tau2 = self.model2(
                next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
            )  # (batch_size*iqn_n,n_actions)

            #   Build IQN target on tau2 quantiles
            outputs_target_tau2 = torch.where(
                done,
                rewards,
                rewards + gammas_pow_nsteps * q__stpo__model2__quantiles_tau2.gather(1, a__tpo__model__reduced_repeated),
            )  # (batch_size*iqn_n, 1)

            #   This is our target
            outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                0, 1
            )  # (batch_size, iqn_n, 1)

            q__st__model__quantiles_tau3, tau3 = self.model(
                state_img_tensor, state_float_tensor, self.iqn_n, tau=None
            )  # (batch_size*iqn_n,n_actions)

            outputs_tau3 = (
                q__st__model__quantiles_tau3.gather(1, actions_n).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
            )  # (batch_size, iqn_n, 1)

            TD_Error = outputs_target_tau2[:, :, None, :] - outputs_tau3[:, None, :, :]
            # (batch_size, iqn_n, iqn_n, 1)    WTF ????????
            # Huber loss, my alternative
            loss = torch.where(
                torch.abs(TD_Error) <= self.iqn_kappa,
                0.5 * TD_Error**2,
                self.iqn_kappa * (torch.abs(TD_Error) - 0.5 * self.iqn_kappa),
            )
            tau3 = tau3.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
            tau3 = tau3[:, None, :, :].expand([-1, self.iqn_n, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
            loss = (
                (torch.where(TD_Error < 0, 1 - tau3, tau3) * loss / self.iqn_kappa).sum(dim=2).mean(dim=1)[:, 0]
            )  # pinball loss # (batch_size, )

            total_loss = torch.sum(loss)  # total_loss.shape=torch.Size([])

            if do_learn:
                self.scaler.scale(total_loss).backward()

                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss = total_loss.detach().cpu()
        self.execution_stream.synchronize()
        return total_loss

    def get_exploration_action(self, img_inputs, float_inputs):
        with torch.no_grad():
            state_img_tensor = img_inputs.unsqueeze(0).to("cuda", memory_format=torch.channels_last, non_blocking=True)
            state_float_tensor = torch.as_tensor(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            q_values = (
                self.model(state_img_tensor, state_float_tensor, self.iqn_k, tau=None, use_fp32=True)[0]
                .cpu()
                .numpy()
                .astype(np.float32)
                .mean(axis=0)
            )
        r = random()

        if r < self.epsilon:
            # Choose a random action
            get_argmax_on = np.random.randn(*q_values.shape)
        elif r < self.epsilon + self.epsilon_boltzmann:
            get_argmax_on = q_values + self.tau_epsilon_boltzmann * np.random.randn(*q_values.shape)
        else:
            get_argmax_on = q_values + ((self.epsilon + self.epsilon_boltzmann) > 0) * self.tau_greedy_boltzmann * np.random.randn(
                *q_values.shape
            )

        action_chosen_idx = np.argmax(get_argmax_on)
        greedy_action_idx = np.argmax(q_values)

        return (
            action_chosen_idx,
            action_chosen_idx == greedy_action_idx,
            np.max(q_values),
            q_values,
        )