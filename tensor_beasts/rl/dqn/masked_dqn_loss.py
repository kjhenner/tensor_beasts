from dataclasses import dataclass
import torch

from tensordict import NestedKey, TensorDictBase, TensorDict
from torchrl.envs import step_mdp
from torchrl.objectives import DQNLoss, distance_loss
from tensordict.nn import dispatch
from torchrl.objectives.utils import _reduce


class MaskedDQNLoss(DQNLoss):
    """A DQNLoss subclass that uses a mask tensor to select valid actions and predictions."""

    @dataclass
    class _AcceptedKeys:
        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "chosen_action_value"
        action_value: NestedKey = "action_value"
        action: NestedKey = "action"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        mask: NestedKey = "mask"

    default_keys = _AcceptedKeys()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_in_keys()  # Update in_keys to include the mask

    def _set_in_keys(self):
        super()._set_in_keys()
        self._in_keys.append(self.tensor_keys.mask)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the DQN loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the DQN loss.

        """
        td_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            if action.ndim != pred_val.ndim:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1)

        if self.double_dqn:
            step_td = step_mdp(td_copy, keep_other=False)
            step_td_copy = step_td.clone(False)
            # Use online network to compute the action
            with self.value_network_params.data.to_module(self.value_network):
                self.value_network(step_td)
                next_action = step_td.get(self.tensor_keys.action)

            # Use target network to compute the values
            with self.target_value_network_params.to_module(self.value_network):
                self.value_network(step_td_copy)
                next_pred_val = step_td_copy.get(self.tensor_keys.action_value)

            if self.action_space == "categorical":
                if next_action.ndim != next_pred_val.ndim:
                    # unsqueeze the action if it lacks on trailing singleton dim
                    next_action = next_action.unsqueeze(-1)
                next_value = torch.gather(next_pred_val, -1, index=next_action)
            else:
                next_value = (next_pred_val * next_action).sum(-1, keepdim=True)
        else:
            next_value = None
        target_value = self.value_estimator.value_estimate(
            td_copy,
            target_params=self.target_value_network_params,
            next_value=next_value,
        ).squeeze(-1)

        with torch.no_grad():
            priority_tensor = (pred_val_index - target_value).pow(2)
            priority_tensor = priority_tensor.unsqueeze(-1)
        if tensordict.device is not None:
            priority_tensor = priority_tensor.to(tensordict.device)

        tensordict.set(
            self.tensor_keys.priority,
            priority_tensor,
            inplace=True,
        )
        mask = tensordict.get(self.tensor_keys.mask)
        loss = distance_loss(pred_val_index[mask], target_value[mask], self.loss_function)
        loss = _reduce(loss, reduction=self.reduction)
        td_out = TensorDict({"loss": loss}, [])
        return td_out
