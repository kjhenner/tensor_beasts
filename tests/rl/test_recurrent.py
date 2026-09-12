"""The differentiable memory carry must match the simulation's own carry.

Small worlds for speed; this is bookkeeping, not ecology.
"""

import pytest
import torch

from tensor_beasts.rl.multiagent import MultiAgentWorldEnv
from tensor_beasts.rl.recurrent import propagate_memory

SIZE = (48, 48)
K = 3


def test_propagated_write_equals_what_the_simulation_carried():
    """Write a random memory, step the world, and check that routing that
    write through the successor map reproduces the memory channels the
    environment actually observes next step, at every cell."""
    env = MultiAgentWorldEnv(size=SIZE, device="cpu", memory_size=K)
    env.reset(seed=0)
    for _ in range(8):
        env.world.update()

    torch.manual_seed(2)
    for _ in range(6):
        write = torch.rand(K, *SIZE) * 2 - 1
        batch = env.step(torch.randint(0, 5, SIZE), memory=write)
        predicted = propagate_memory(write, batch.successor, batch.acted, batch.reproduced, batch.done)

        observation = env._build_observation()
        actual = torch.stack([observation[env.channel_names.index(f"memory/{k}")] for k in range(K)])
        assert torch.allclose(predicted, actual, atol=1e-6), "recomputed carry disagrees with the simulation"


def test_dead_individuals_write_goes_nowhere():
    write = torch.ones(K, 4, 4)
    acted = torch.zeros(4, 4, dtype=torch.bool)
    acted[1, 1] = True
    successor = torch.arange(16).reshape(4, 4)
    successor[1, 1] = 6  # would move to (1, 2)
    done = acted.clone()  # it died
    out = propagate_memory(write, successor, acted, torch.zeros(4, 4, dtype=torch.bool), done)
    assert torch.all(out == 0)


def test_offspring_get_a_copy_and_parent_moves_on():
    write = torch.zeros(K, 4, 4)
    write[:, 1, 1] = torch.tensor([0.2, -0.4, 0.6])
    acted = torch.zeros(4, 4, dtype=torch.bool)
    acted[1, 1] = True
    successor = torch.arange(16).reshape(4, 4)
    successor[1, 1] = 5  # moved to (1, 0)... flat 5 is (1, 1)? use (0, 1) = 1
    successor[1, 1] = 1
    reproduced = acted.clone()
    out = propagate_memory(write, successor, acted, reproduced)
    assert torch.allclose(out[:, 0, 1], write[:, 1, 1]), "parent carried its memory to the new cell"
    assert torch.allclose(out[:, 1, 1], write[:, 1, 1]), "offspring inherited a copy at the origin"
    assert float(out.abs().sum()) == float(write[:, 1, 1].abs().sum()) * 2


def test_gradient_flows_from_read_back_to_write():
    """The reason this function exists: a loss on the next step's read must
    produce a gradient on this step's write."""
    write = (torch.rand(K, 4, 4) * 2 - 1).requires_grad_(True)
    acted = torch.zeros(4, 4, dtype=torch.bool)
    acted[2, 2] = True
    successor = torch.arange(16).reshape(4, 4)
    successor[2, 2] = 10  # moves to (2, 2)->(2, 2)? flat 10 = (2, 2); move to flat 9 = (2, 1)
    successor[2, 2] = 9
    read = propagate_memory(write, successor, acted, torch.zeros(4, 4, dtype=torch.bool))
    read[:, 2, 1].sum().backward()
    assert torch.all(write.grad[:, 2, 2] == 1.0), "gradient reached the cell that wrote"
    assert float(write.grad.abs().sum()) == float(K), "and nowhere else"



def _collect(memory_size, steps=6, size=32):
    from tensor_beasts.rl.ppo import PPOConfig
    from tensor_beasts.rl.trainer import Trainer, TrainerConfig

    torch.manual_seed(0)
    trainer = Trainer(
        TrainerConfig(size=size, arch="conv", arch_kwargs={"hidden_channels": 8}, memory_size=memory_size,
                      warmup_steps=3, total_world_steps=0, eval_interval=0, checkpoint_interval=0, device="cpu"),
        PPOConfig(),
    )
    rollout, _ = trainer.collect(steps)
    return trainer, rollout


def test_recomputed_reads_match_stored_reads_with_unchanged_weights():
    """The consistency check behind recurrent training: replaying the segment
    with the same weights must reproduce, at every step, the memory channels
    the environment stored, up to float16 storage rounding."""
    from tensor_beasts.rl.ppo import PPO

    trainer, rollout = _collect(memory_size=2)
    network, k = trainer.network, 2
    read_prev = None
    with torch.no_grad():
        for t in range(rollout.steps):
            stored = rollout.observation[t].float()
            if read_prev is not None:
                assert torch.allclose(read_prev, stored[-k:], atol=2e-3), f"read at step {t} diverged from storage"
                observation = torch.cat([stored[:-k], read_prev])
            else:
                observation = stored
            heads = PPO._heads(network, observation.unsqueeze(0), rollout.action[t : t + 1])
            from tensor_beasts.rl.recurrent import propagate_memory
            read_prev = propagate_memory(
                heads["memory"][0], rollout.successor[t], rollout.acted[t], rollout.reproduced[t], rollout.done[t]
            )


def test_recurrent_update_reaches_the_write_head_and_stage_one_does_not():
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    trainer, rollout = _collect(memory_size=2)
    network = trainer.network

    def head_gradient(window):
        ppo = PPO(PPOConfig(recurrent_window=window, epochs=1, minibatch_steps=rollout.steps),
                  value_normalizer=trainer.value_normalizer)
        network.zero_grad(set_to_none=True)
        # Hook the head so we see its gradient even though the optimizer steps.
        seen = {}
        def record(grad):
            seen["grad"] = seen.get("grad", 0.0) + grad.detach().abs().sum().item()

        handle = network.memory_head.weight.register_hook(record)
        result = ppo.update(network, rollout, torch.optim.Adam(network.parameters(), lr=1e-9))
        handle.remove()
        return seen.get("grad", 0.0), result

    stage_one, _ = head_gradient(window=0)
    recurrent, result = head_gradient(window=rollout.steps)
    assert stage_one == 0.0, "without recurrence nothing differentiates the write"
    assert recurrent > 0.0, "with recurrence the next step's loss reaches the write head"
    assert result["recurrent_window"] == rollout.steps
    assert 0.0 <= result["memory_write_abs_mean"] <= 1.0


def test_first_recurrent_epoch_has_unit_ratio():
    """Standard PPO check, and here it also proves the recomputed reads are
    the ones the behaviour policy actually saw."""
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    trainer, rollout = _collect(memory_size=2)
    ppo = PPO(PPOConfig(recurrent_window=3, epochs=1), value_normalizer=trainer.value_normalizer)
    result = ppo.update(trainer.network, rollout, torch.optim.Adam(trainer.network.parameters(), lr=1e-9))
    assert result["approx_kl"] < 1e-4
    assert result["ratio_max_deviation"] < 5e-2


def test_a_task_that_needs_memory_is_only_learned_recurrently():
    """The reason for all of this. The target at step t is the argmax of five
    observation channels the individual saw at step t-1, carried through its
    move. A policy can only match it by writing what it saw and reading it
    back. Stage one cannot learn the write; recurrent training must."""
    from tensor_beasts.rl.networks import build_network
    from tensor_beasts.rl.ppo import PPO, PPOConfig
    from tensor_beasts.rl.rollout import Rollout

    torch.manual_seed(0)
    T, size, K = 10, 8, 5
    cells = size * size
    obs_channels = 6
    observation = torch.rand(T, obs_channels + K, size, size)
    observation[:, obs_channels:] = 0.0  # stored memory reads: blank
    acted = torch.ones(T, size, size, dtype=torch.bool)
    done = torch.zeros(T, size, size, dtype=torch.bool)
    reproduced = torch.zeros(T, size, size, dtype=torch.bool)
    successor = torch.stack([torch.randperm(cells).reshape(size, size) for _ in range(T)])

    rule_action = torch.zeros(T, size, size, dtype=torch.long)
    for t in range(1, T):
        carried = propagate_memory(observation[t - 1, :5], successor[t - 1], acted[t - 1], reproduced[t - 1])
        rule_action[t] = carried.argmax(dim=0)

    def make_rollout():
        return Rollout(
            observation=observation.half(), acted=acted, action=torch.randint(0, 5, (T, size, size)),
            log_prob=torch.full((T, size, size), -1.6094), value=torch.zeros(T, size, size),
            reward=torch.zeros(T, size, size), done=done, successor=successor,
            advantage=torch.zeros(T, size, size), ret=torch.zeros(T, size, size),
            rule_action=rule_action, reproduced=reproduced,
        )

    def agreement_after_training(window):
        torch.manual_seed(1)
        network = build_network("linear", obs_channels + K, memory_size=K)
        ppo = PPO(PPOConfig(recurrent_window=window, epochs=1, minibatch_steps=T, imitation_coef=1.0,
                            imitation_temperature=0.0, imitation_target_conformance=1.01, entropy_coef=0.0,
                            value_coef=0.0, learning_rate=0.05))
        optimizer = torch.optim.Adam(network.parameters(), lr=0.05)
        for _ in range(120):
            result = ppo.update(network, make_rollout(), optimizer)
        # Score agreement at steps >= 1 with the recomputed reads, as deployment would see them.
        with torch.no_grad():
            read_prev, agree, count = None, 0, 0
            for t in range(T):
                obs = observation[t].clone()
                if read_prev is not None:
                    obs[obs_channels:] = read_prev
                logits, _ = network(obs.unsqueeze(0))
                if t >= 1:
                    agree += int((logits.argmax(1)[0] == rule_action[t]).sum()); count += cells
                out = network.forward_all(obs.unsqueeze(0))
                read_prev = propagate_memory(out["memory"][0], successor[t], acted[t], reproduced[t])
        return agree / count

    stage_one = agreement_after_training(window=0)
    recurrent = agreement_after_training(window=T)
    assert recurrent > stage_one + 0.25, f"recurrent {recurrent:.3f} vs stage one {stage_one:.3f}"
    assert recurrent > 0.7



def test_recurrent_diagnostics_are_not_diluted_by_window_gradient_norms():
    """A fresh policy's first update must report entropy near ln(5), and the
    recurrent path must report the same diagnostics as the plain path on the
    same rollout with the same weights. The bug this pins: the per-window
    gradient norm was added to the shared accumulator with the window's whole
    weight, which halved every other diagnostic, including the conformance the
    anchor's cross-fade is keyed on."""
    import math
    from tensor_beasts.rl.ppo import PPO, PPOConfig

    trainer, rollout = _collect(memory_size=2)
    plain = PPO(PPOConfig(recurrent_window=0, epochs=1, minibatch_steps=rollout.steps),
                value_normalizer=trainer.value_normalizer)
    recurrent = PPO(PPOConfig(recurrent_window=3, epochs=1, minibatch_steps=rollout.steps),
                    value_normalizer=trainer.value_normalizer)
    frozen = torch.optim.Adam(trainer.network.parameters(), lr=1e-12)
    a = plain.update(trainer.network, rollout, frozen)
    b = recurrent.update(trainer.network, rollout, frozen)
    assert abs(b["entropy"] - math.log(5)) < 0.05, f"first-update entropy {b['entropy']:.3f}"
    for key in ("entropy", "policy_loss", "value_loss", "approx_kl"):
        assert b[key] == pytest.approx(a[key], abs=2e-2), key
    assert "grad_norm" in b and b["grad_norm"] > 0
