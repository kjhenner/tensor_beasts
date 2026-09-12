"""Drive a live World with a trained policy checkpoint.

This is the bridge between training and watching. train_rl.py saves a
checkpoint holding the network weights plus the trainer config that built
them; :class:`LearnedController` rebuilds that network, attaches the
observation encoding to an existing World, and produces the per-step action
TensorDict that ``World.update`` accepts. The interactive viewer uses it via
``python -m tensor_beasts --policy <checkpoint>``.
"""

from pathlib import Path
from typing import Optional

import torch
from tensordict import TensorDict

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv
from tensor_beasts.rl.networks import build_network
from tensor_beasts.world import World

NUM_ACTIONS = 5


class LearnedController:
    """Chooses every controlled individual's direction from a checkpoint.

    Args:
        world: The live world to observe and act in.
        checkpoint: Path to a train_rl.py checkpoint.
        entity_name: Which entity the policy controls; defaults to whatever
            the checkpoint was trained on.
        deterministic: Take the most likely direction rather than sampling.
            Sampling is what training and evaluation use, so it is the default;
            deterministic is useful for seeing the policy's clearest intent.
        device: Where to run the network. Defaults to the world's device.
    """

    def __init__(
        self,
        world: World,
        checkpoint: Path,
        entity_name: Optional[str] = None,
        deterministic: bool = False,
        device: Optional[torch.device] = None,
    ):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        trainer_config = payload["trainer_config"]
        self.entity_name = entity_name or trainer_config.get("entity", "Herbivore")
        self.deterministic = deterministic

        self.env = MultiAgentWorldEnv.attach(world, self.entity_name)
        self.device = device or self.env.device

        expected = payload.get("observation_channels", self.env.observation_channels)
        if expected != self.env.observation_channels:
            raise ValueError(
                f"Checkpoint expects {expected} observation channels but this world "
                f"produces {self.env.observation_channels}. The perception config "
                "differs from the one the policy was trained on."
            )

        self.network = build_network(
            trainer_config["arch"],
            self.env.observation_channels,
            **(trainer_config.get("arch_kwargs") or {}),
        )
        self.network.load_state_dict(payload["network"])
        self.network.to(self.device).eval()

        self.trained_world_steps = payload.get("world_steps")
        self.arch = trainer_config["arch"]

    @torch.no_grad()
    def action(self) -> TensorDict:
        """Directions for every cell, as the TensorDict World.update expects."""
        observation = self.env._build_observation().to(self.device)
        logits, _ = self.network(observation.unsqueeze(0))
        if self.deterministic:
            direction = logits.argmax(dim=1)
        else:
            flat = torch.log_softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, NUM_ACTIONS)
            direction = torch.multinomial(flat.exp(), 1).reshape(1, *self.env.size)
        return TensorDict({self.entity_name: direction.squeeze(0).to(self.env.device)}, batch_size=[])

    def describe(self) -> str:
        steps = f", trained for {self.trained_world_steps} world steps" if self.trained_world_steps else ""
        mode = "deterministic" if self.deterministic else "sampled"
        return f"learned {self.arch} policy on {self.entity_name} ({mode}{steps})"
