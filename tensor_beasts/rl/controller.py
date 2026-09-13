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
    """Chooses every controlled individual's action from a checkpoint.

    A checkpoint trained with a metabolic head drives both levers: the action
    sent to the world is then a mapping with ``direction`` and
    ``metabolic_rate``, which ``Animal.update`` clamps to what each
    individual's biomass allows. A direction-only checkpoint sends the bare
    direction tensor, exactly as before.

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

        # Recorded explicitly by newer checkpoints; older ones predate the
        # metabolic head and mean zero.
        self.num_metabolic_levels = int(
            payload.get("num_metabolic_levels", trainer_config.get("metabolic_levels", 0))
        )
        self.memory_size = int(payload.get("memory_size", trainer_config.get("memory_size", 0)))

        self.env = MultiAgentWorldEnv.attach(
            world, self.entity_name, num_metabolic_levels=self.num_metabolic_levels
        )
        self.device = device or self.env.device

        expected = payload.get("observation_channels", self.env.observation_channels)
        if expected != self.env.observation_channels:
            raise ValueError(
                f"Checkpoint expects {expected} observation channels but this world "
                f"produces {self.env.observation_channels}. The perception config "
                "differs from the one the policy was trained on."
            )

        if self.env.memory_size != self.memory_size:
            raise ValueError(
                f"Checkpoint carries {self.memory_size} memory channels but the world's "
                f"{self.entity_name} has {self.env.memory_size}. Call "
                "apply_checkpoint_requirements(config, checkpoint) before building the world."
            )

        self.network = build_network(
            trainer_config["arch"],
            self.env.observation_channels,
            num_metabolic_levels=self.num_metabolic_levels,
            memory_size=self.memory_size,
            **(trainer_config.get("arch_kwargs") or {}),
        )
        self.network.load_state_dict(payload["network"])
        self.network.to(self.device).eval()

        self.trained_world_steps = payload.get("world_steps")
        self.arch = trainer_config["arch"]

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """One categorical per cell from ``(1, K, H, W)`` logits, as ``(H, W)``."""
        if self.deterministic:
            return logits.argmax(dim=1).squeeze(0)
        flat = torch.log_softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        return torch.multinomial(flat.exp(), 1).reshape(*self.env.size)

    @torch.no_grad()
    def action(self) -> TensorDict:
        """The action for every cell, as the TensorDict World.update expects.

        Direction-only checkpoints map the entity to a bare (H, W) direction
        tensor. Two-lever checkpoints map it to a nested TensorDict with
        ``direction`` and ``metabolic_rate``; the level chosen by the network
        is turned into a rate here, and the simulation applies the biomass cap.
        """
        observation = self.env._build_observation().to(self.device)
        out = self.network.forward_all(observation.unsqueeze(0))
        direction = self._sample(out["logits"]).to(self.env.device)
        if "metabolic_logits" not in out and "memory" not in out:
            return TensorDict({self.entity_name: direction}, batch_size=[])
        fields = {"direction": direction}
        if "metabolic_logits" in out:
            level = self._sample(out["metabolic_logits"]).to(self.env.device)
            fields["metabolic_rate"] = self.env.metabolic_level_to_rate(level)
        if "memory" in out:
            # Network layout (K, H, W) to the feature's (H, W, K).
            fields["memory"] = out["memory"].squeeze(0).permute(1, 2, 0).to(self.env.device)
        return TensorDict({self.entity_name: TensorDict(fields, batch_size=[])}, batch_size=[])

    def describe(self) -> str:
        steps = f", trained for {self.trained_world_steps} world steps" if self.trained_world_steps else ""
        mode = "deterministic" if self.deterministic else "sampled"
        levers = (
            f", movement + {self.num_metabolic_levels}-level metabolism"
            if self.num_metabolic_levels
            else ", movement only"
        )
        return f"learned {self.arch} policy on {self.entity_name} ({mode}{levers}{steps})"


def apply_checkpoint_requirements(config, checkpoint: Path, entity_name: Optional[str] = None) -> None:
    """Make a world config match what a checkpoint needs, before the world exists.

    A checkpoint trained with learned memory expects the controlled entity to
    carry that many memory channels, which is an entity feature fixed at world
    construction. The viewer builds its world from a plain config, so this sets
    the feature width from the checkpoint. Harmless for checkpoints without
    memory.
    """
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    trainer_config = payload.get("trainer_config", {})
    entity = entity_name or trainer_config.get("entity", "Herbivore")
    memory_size = int(payload.get("memory_size", trainer_config.get("memory_size", 0)))
    if memory_size > 0:
        config.world.entities[entity].memory = {"size": memory_size}
