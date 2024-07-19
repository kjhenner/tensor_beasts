import tempfile
import time
import copy

import hydra
import torch.nn
import torch.optim
import tqdm
from tensordict.nn import TensorDictSequential
from torchrl._utils import logger as torchrl_logger

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import HardUpdate
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger

from tensor_beasts.rl.dqn.masked_dqn_loss import MaskedDQNLoss
from tensor_beasts.rl.dqn.utils import eval_model, make_dqn_model, make_env


@hydra.main(config_path="", config_name="tensor_beasts_config", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    device = cfg.device
    if device in ("", None, "auto"):
        if torch.cuda.is_available():
            cfg.device = "cuda:0"
        elif torch.backends.mps.is_available():
            cfg.device = "mps"
        else:
            cfg.device = "cpu"
    device = torch.device(cfg.device)

    total_frames = cfg.collector.total_frames
    frames_per_batch = cfg.collector.frames_per_batch
    init_random_frames = cfg.collector.init_random_frames
    test_interval = cfg.logger.test_interval

    # Make the components
    model = make_dqn_model(cfg)
    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=model.spec,
    )
    model_explore = TensorDictSequential(
        model,
        greedy_module,
    ).to(device)

    # Create the collector
    collector = SyncDataCollector(
        create_env_fn=make_env(device, cfg.env),
        policy=model_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        init_random_frames=init_random_frames,
        return_same_td=False
    )

    # Create the replay buffer
    if cfg.buffer.scratch_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        scratch_dir = tempdir.name
    else:
        scratch_dir = cfg.buffer.scratch_dir
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyMemmapStorage(
            max_size=cfg.buffer.buffer_size,
            scratch_dir=scratch_dir,
        ),
        batch_size=cfg.buffer.batch_size,
    )

    # Create the loss module
    loss_module = MaskedDQNLoss(
        value_network=model,
        loss_function="l2",
        delay_value=True,
        action_space=model.spec["action"],
    )
    loss_module.set_keys(done="terminated", terminated="terminated")
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("DQN", f"TensorBeasts_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="dqn",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "mode": cfg.logger.mode,
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Create the test environment
    test_env = make_env(cfg.device, cfg.env)
    if cfg.logger.video:
        test_env.insert_transform(
            0,
            VideoRecorder(
                logger, tag=f"rendered/{cfg.env.env_name}", in_keys=["rgb_array"]
            ),
        )
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    sampling_start = time.time()
    num_updates = cfg.loss.num_updates
    max_grad = cfg.optim.max_grad_norm
    num_test_episodes = cfg.logger.num_test_episodes
    q_losses = torch.zeros(num_updates, device=device)
    pbar = tqdm.tqdm(total=total_frames)

    cumulative_rewards = 0
    eval_count = 0
    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        collected_frames += current_frames
        greedy_module.step(current_frames)
        replay_buffer.extend(data)

        # Get and log training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_length_mean = episode_length.sum().item() / len(episode_length)
            log_info.update(
                {
                    "train/episode_reward": episode_reward_mean,
                    "train/episode_length": episode_length_mean,
                }
            )

        if collected_frames < init_random_frames:
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, step=collected_frames)
            continue

        # optimization steps
        training_start = time.time()
        for j in range(num_updates):

            sampled_tensordict = replay_buffer.sample()
            sampled_tensordict = sampled_tensordict.to(device)
            batch_size = sampled_tensordict["action"].shape[0]
            num_actions = sampled_tensordict["action"].shape[-2]

            # sampled_tensordict['mask'] = (sampled_tensordict["observation"][..., 4] > 0).flatten(start_dim=1)
            sampled_tensordict["mask"] = sampled_tensordict["mask"].flatten(start_dim=1)

            sampled_tensordict['next', 'reward'] = sampled_tensordict['next', 'reward'].view(batch_size, 1, 1).expand(batch_size, num_actions, 1)
            sampled_tensordict['next', 'done'] = sampled_tensordict['next', 'done'].view(batch_size, 1, 1).expand(batch_size, num_actions, 1)
            sampled_tensordict['next', 'terminated'] = sampled_tensordict['next', 'terminated'].view(batch_size, 1, 1).expand(batch_size, num_actions, 1)


            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     list(loss_module.parameters()), max_norm=max_grad
            # )
            optimizer.step()
            target_net_updater.step()
            q_losses[j].copy_(q_loss.detach())

        training_time = time.time() - training_start

        # Get and log q-values, loss, epsilon, sampling time and training time
        log_info.update(
            {
                "train/q_values_mean": (data["action_value"] * data["action"]).mean().item() / frames_per_batch,
                "train/q_loss": q_losses.mean().item(),
                "train/epsilon": greedy_module.eps,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/action_freqencies": {int(value): int(count) for value, count in zip(*data["action"].argmax(-1).unique(return_counts=True))}
            }
        )

        # Get and log evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            prev_test_frame = ((i - 1) * frames_per_batch) // test_interval
            cur_test_frame = (i * frames_per_batch) // test_interval
            final = current_frames >= collector.total_frames
            if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
                model.eval()
                eval_start = time.time()
                test_rewards = eval_model(
                    model, test_env, num_episodes=num_test_episodes
                )
                eval_time = time.time() - eval_start
                cumulative_rewards += test_rewards
                eval_count += num_test_episodes
                log_info.update(
                    {
                        "eval/reward": test_rewards,
                        "eval/mean_reward": cumulative_rewards / eval_count,
                        "eval/eval_time": eval_time,
                    }
                )
                model.train()

        # Log all the information
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_(copy.deepcopy(model.state_dict()))
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
