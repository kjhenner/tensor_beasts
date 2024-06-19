from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey
from torch import nn
from torch.nn import functional as F
from torchrl.data.tensor_specs import TensorSpec

from torchrl.envs.utils import step_mdp
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _GAMMA_LMBDA_DEPREC_ERROR, _reduce


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
        gamma: float = 0.5,
    ) -> None:
        super().__init__()
        self._in_keys = None
        value_network = ensure_tensordict_compatible(
            module=value_network,
            wrapper_type=QValueActor,
        )

        self.convert_to_functional(
            value_network,
            "value_network"
        )

        self.value_network_in_keys = value_network.in_keys

        # Currently bypassing this separate value estimator thing as both the values and actions
        # are being computed by the same network. Could be worth revisiting this to be consistent
        # with the rest of TorchRL!
        # if gamma is not None:
        #     raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

        self.gamma = gamma

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

    def _calculate_quantile_huber_loss(self, current_q, target_q, quantiles):
        """Calculates the quantile Huber loss."""
        td_errors = target_q - current_q

        huber_loss = F.huber_loss(current_q, target_q, reduction='none', delta=1.0)

        quantile_loss = (quantiles - (td_errors.detach() < 0).float()).abs() * huber_loss
        return quantile_loss.mean()

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the IQN loss given a tensordict sampled from the replay buffer.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network ("observations", "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the IQN loss.

        """
        current_td = tensordict.clone(False)

        with self.value_network_params.to_module(self.value_network):
            self.value_network(current_td)

        # Indices of selected actions (N, W*H)
        current_action_idx = tensordict.get(self.tensor_keys.action)

        # (N, num_quantiles, W*H, num_actions)
        pred_Q = current_td.get(self.tensor_keys.quantile)
        tau = current_td.get(self.tensor_keys.tau)

        N = tensordict.get(self.tensor_keys.action).shape[0]
        num_quantiles = tensordict.get(self.tensor_keys.tau).shape[1]

        current_action_idx = current_action_idx.unsqueeze(-1).expand(N, num_quantiles, current_action_idx.shape[-1], 1)

        pred_Q = pred_Q.gather(3, current_action_idx).squeeze(-1)

        step_td = step_mdp(current_td, keep_other=False, exclude_reward=False)
        step_td_copy = step_td.clone(False)

        # Use online network to compute the action
        with self.value_network_params.data.to_module(self.value_network):
            self.value_network(step_td)

        next_action_idx = step_td.get(self.tensor_keys.action)
        next_action_idx = next_action_idx.unsqueeze(1).unsqueeze(-1).expand(N, num_quantiles, next_action_idx.shape[-1], 1)

        # Use target network to compute the values
        with self.target_value_network_params.to_module(self.value_network):
            self.value_network(step_td_copy)
            next_Q = step_td_copy.get(self.tensor_keys.quantile)

        next_Q = next_Q.gather(3, next_action_idx)

        reward = step_td.get(self.tensor_keys.reward)

        done = step_td.get(self.tensor_keys.done)

        # The target Q is the observed reward plus the discounted next Q value, masked by the done flag
        target_Q = reward + (self.gamma * next_Q.squeeze(1).squeeze(-1)) * (~ done)
        target_Q = target_Q.unsqueeze(1)

        loss = self._calculate_quantile_huber_loss(pred_Q, target_Q, tau)
        loss = _reduce(loss, reduction="mean")
        td_out = TensorDict({"loss": loss}, [])
        return td_out
