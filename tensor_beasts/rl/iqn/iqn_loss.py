from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey
from torch import nn
from torch.nn import functional as F

from torchrl.envs.utils import step_mdp
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce


def calculate_quantile_huber_loss(current_q, target_q, tau):
    """
    Calculates the quantile Huber loss.

    Intuition for quantile loss:
    If squared error gives rise to a mean (balance of magnitudes above and below target)
    And absolute error gives rise to a median (balance between number of data points above and below target),
    Absolute error is like quantile loss with alpha=0.5 for alpha in [0, 1].
    Quantile loss gives rise to a quantile defined by other values of alpha.
    """
    td_errors = target_q - current_q

    huber_loss = F.huber_loss(current_q, target_q, reduction='none', delta=1.0)

    quantile_loss = (tau - (td_errors.detach() < 0).float()).abs() * huber_loss
    return quantile_loss.mean()


class IqnLoss(LossModule):
    """The IQN Loss class.

    Args:
        value_network (QValueActor or nn.Module): a Q value operator.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
            quantile (NestedKey): The key in the input TensorDict where the quantile is expected.
                Defaults to ``"quantile"``.
            tau (NestedKey): The key in the input TensorDict where the tau is expected.
                Defaults to ``"tau"``.
        """

        action: NestedKey = "action"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        quantile: NestedKey = "quantile"
        tau: NestedKey = "tau"

    default_keys = _AcceptedKeys()
    out_keys = ["loss"]

    def __init__(
        self,
        value_network: Union[QValueActor, nn.Module],
        *,
        gamma: float = 0.99,
        num_quantiles: int = 16,
    ) -> None:
        super().__init__()
        self._in_keys = None
        value_network = ensure_tensordict_compatible(
            module=value_network,
            wrapper_type=QValueActor,
        )

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=True
        )

        self.value_network_in_keys = value_network.in_keys

        # Currently bypassing separate value estimator as both the values and actions
        # are being computed by the same network. Could be worth revisiting this to be consistent
        # with the rest of TorchRL!

        # if gamma is not None:
        #     raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

        self.gamma = gamma
        self.num_quantiles = num_quantiles

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.value_network.in_keys,
            *[("next", key) for key in self.value_network.in_keys],
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the IQN loss given a tensordict sampled from the replay buffer.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network ("observations", "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the IQN loss.

        """
        N = tensordict.get(self.tensor_keys.action).shape[0]

        # Run the target model
        with self.target_value_network_params.to_module(self.value_network):
            next_td = self.value_network(
                step_mdp(tensordict, keep_other=False, exclude_reward=False)
            )
        action_idx_next = next_td.get(self.tensor_keys.action)
        Q_targets_next = next_td.get(self.tensor_keys.quantile).detach()

        # TODO: Do we need to transpose this to gather correctly?
        Q_targets_next = Q_targets_next.gather(
            3,
            action_idx_next.unsqueeze(1).expand(N, self.num_quantiles, -1).unsqueeze(-1)
        ).squeeze(-1)

        reward = next_td.get(self.tensor_keys.reward).unsqueeze(-1).expand_as(Q_targets_next)
        next_done = (~ next_td.get(self.tensor_keys.done)).unsqueeze(-1).expand_as(Q_targets_next)
        Q_targets = reward + self.gamma * next_done * Q_targets_next

        # Run the online model
        with self.value_network_params.to_module(self.value_network):
            current_td = self.value_network(tensordict.clone())
        action_idx = current_td.get(self.tensor_keys.action)
        tau = current_td.get(self.tensor_keys.tau)
        Q_expected = current_td.get(self.tensor_keys.quantile)
        Q_expected = Q_expected.gather(
            3, action_idx.unsqueeze(1).expand(N, self.num_quantiles, -1).unsqueeze(-1)
        ).squeeze(-1)

        loss = calculate_quantile_huber_loss(Q_expected, Q_targets, tau)
        td_out = TensorDict({"loss": loss})
        return td_out
