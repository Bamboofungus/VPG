import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from torch.types import Number
import wandb
import hydra
from omegaconf import OmegaConf
import numpy as np
from datetime import datetime
import os
from typing import Tuple
from collections import namedtuple, deque
from tqdm import tqdm

if not torch.cuda.is_available():
    print("GPU not detected, falling back to CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    policy,
    f_approximator,
    policy_optimizer,
    f_approximator_optimizer,
    epoch,
    hyperparameters,
    checkpoint_directory,
    wandb_runname=None,
):
    checkpoint = {
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "f_approximator_state_dict": f_approximator.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "f_approximator_optimizer_state_dict": f_approximator_optimizer.state_dict(),
        "hyperparameters": OmegaConf.to_container(
            hyperparameters, resolve=True, throw_on_missing=True
        ),
        "wandb_runname": wandb_runname,
    }
    filepath = f"{checkpoint_directory}{datetime.now().timestamp()}"
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")


BufferEntry = namedtuple(
    "BufferEntry",
    ["observation", "action", "reward", "value", "log_prob", "rtg", "advantage"],
)


class TrajectoryBuffer:
    """
    Buffer for storing data from policy interacting with environment for use in training at end of epochs
    """

    def __init__(self, gamma=0.99, adv_norm=True):
        self.buffer = []
        self.curr_trajectory_start = 0
        self.gamma = gamma
        self.adv_norm = adv_norm

    def end_trajectory(self, trajectory_end, final_value=0):
        curr_rtg = final_value
        # calculate return-to-gos/advantages
        for t in reversed(range(self.curr_trajectory_start, trajectory_end + 1)):
            curr_rtg = self.buffer[t].reward + curr_rtg * self.gamma
            self.buffer[t] = BufferEntry(
                observation=self.buffer[t].observation,
                action=self.buffer[t].action,
                reward=self.buffer[t].reward,
                value=self.buffer[t].value,
                log_prob=self.buffer[t].log_prob,
                rtg=curr_rtg,
                advantage=curr_rtg - self.buffer[t].value,
            )

        self.curr_trajectory_start = trajectory_end + 1

    def get_trajectories(self):
        # separate observations, actions, etc. into different tensors
        observations, actions, rewards, values, log_probs, rtgs, advantages = zip(
            *self.buffer
        )
        observations = torch.tensor(np.array(observations), dtype=torch.float32).to(
            device
        )
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(device)
        log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(device)
        rtgs = torch.tensor(np.array(rtgs), dtype=torch.float32).to(device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(device)

        # normalize advantages
        if self.adv_norm:
            advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-8
            )  # 1e-8 prevents division by zero errors

        self.buffer = []
        self.curr_trajectory_start = 0

        return observations, actions, rewards, values, log_probs, rtgs, advantages


# used to initially center action dist around 0
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNN(nn.Module):
    """
    Fully connected NN w/ Epsilon greedy, only works for discrete actions spaces

    Outputs a probability for each action from the passed state which can then be sampled
    to choose the next action as well as log_prob for calculating gradients
    """

    def __init__(
        self,
        input_size=8,
        hidden_size=256,
        output_size=4,
        lr=1e-4,
        epsilon=0.05,
        ent_coeff=0.01,
    ):
        super(PolicyNN, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, output_size)),
            nn.Softmax(dim=-1),  # Normalize probabilties to sum to one
        )
        self.epsilon = epsilon
        self.ent_coeff = ent_coeff
        self.optimizer = Adam(self.parameters(), lr)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(device)
        action_probs = self.network(x).squeeze()
        return Categorical(action_probs)

    def act(self, observation) -> Tuple[Number, float]:
        with torch.no_grad():
            action_dist = self.forward(observation)
            if self.training and np.random.rand() < self.epsilon:
                action = torch.tensor(np.random.randint(0, 3)).to(device)
            else:
                action = action_dist.sample()
            log_prob = action_dist.log_prob(action).item()
        return action.item(), log_prob

    def loss(self, observations, actions, advantages, old_log_probs):
        action_dists = self.forward(observations)
        log_probs = action_dists.log_prob(actions)

        approx_kl = (old_log_probs - log_probs).mean().item()
        ent = action_dists.entropy().mean().item()
        policy_metadata = dict(kl=approx_kl, ent=ent)

        # important: negate this loss since it's meant for gradient ascent
        return (
            -(log_probs * advantages).mean() + (ent * self.ent_coeff),
            policy_metadata,
        )


class ValueNN(nn.Module):
    """
    Outputs the estimated value of a passed in state which can then be used in calculating advantage
    """

    def __init__(self, input_size=8, hidden_size=256, output_size=1, lr=1e-4):
        super(ValueNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.optimizer = Adam(self.parameters(), lr)

    def forward(self, x) -> float:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(device)
        return self.network(x).item()

    def loss(self, observations: torch.Tensor, rtgs: torch.Tensor):
        return ((self.network(observations) - rtgs) ** 2).mean()


@hydra.main(config_path="config", config_name="vpg")
def train(hyperparams):
    print("Training VPG with hyperparameters:", hyperparams)

    # load hyperparameters from hydra
    total_epochs = hyperparams.epochs
    episode_steps_per_epoch = hyperparams.episode_steps_per_epoch
    gamma = (
        hyperparams.gamma
    )  # discount rate for rewards during returns-to-go calculation
    normalize_advantages = hyperparams.normalize_advantages
    epsilon = hyperparams.epsilon
    entropy_coefficient = hyperparams.entropy_coeff
    seed = hyperparams.seed
    torch_deterministic = hyperparams.torch_deterministic

    policy_layers = hyperparams.policy.model.layers
    policy_orthogonal_init = hyperparams.policy.model.use_orthogonal_init
    policy_learning_rate = hyperparams.policy.optimizer.learning_rate
    anneal_policy_lr = hyperparams.policy.optimizer.anneal_lr

    value_layers = hyperparams.value.model.layers
    value_orthogonal_init = hyperparams.value.model.use_orthogonal_init
    value_learning_rate = hyperparams.value.optimizer.learning_rate
    anneal_value_lr = hyperparams.value.optimizer.anneal_lr
    value_steps_per_epoch = hyperparams.value.optimizer.steps_per_epoch

    # logging config
    LOGGING = True
    CHECKPOINT_INTERVAL = 50
    RUN_NAME = f"lunarlander_vpg_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}/"
    CHECKPOINT_DIRECTORY = "./models/"
    RECORD_VIDEO = True
    RECORD_EPISODE_INTERVAL = 250
    VIDEO_DIRECTORY = "./videos/"

    # TRY NOT TO MODIFY: seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)
    observation, _ = env.reset(seed=seed)

    if RECORD_VIDEO:
        os.makedirs(os.path.dirname(VIDEO_DIRECTORY + RUN_NAME), exist_ok=True)
        env = RecordVideo(
            env,
            VIDEO_DIRECTORY + RUN_NAME,
            episode_trigger=lambda ep: ep != 0 and ep % RECORD_EPISODE_INTERVAL == 0,
        )

    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="lunarlander_vpg",
        # Track hyperparameters and run metadata
        mode="online" if LOGGING else "disabled",
        config=OmegaConf.to_container(hyperparams, resolve=True, throw_on_missing=True),
    )

    os.makedirs(os.path.dirname(CHECKPOINT_DIRECTORY + RUN_NAME), exist_ok=True)

    policy = PolicyNN(
        lr=policy_learning_rate, epsilon=epsilon, ent_coeff=entropy_coefficient
    ).to(device)
    # function approximation must be used here since LunarLander's observation_space is continous
    f_approximator = ValueNN(lr=value_learning_rate).to(device)
    replay_buffer = TrajectoryBuffer(gamma, normalize_advantages)

    rew_running_mean_buffer = deque(maxlen=100)
    total_episodes = 0

    # initialize per episode variables
    rewards_per_episode = []
    values_per_episode = []
    episode_length = 0
    terminated = truncated = False

    for epoch in range(1, total_epochs + 1):
        print(f"Running epoch #{epoch} out of {total_epochs}")

        # decay learning rate every epoch if enabled
        frac = 1.0 - (epoch - 1.0) / total_epochs
        if anneal_policy_lr:
            policy_lrnow = frac * policy_learning_rate
            policy.optimizer.param_groups[0]["lr"] = policy_lrnow
        if anneal_value_lr:
            f_approximator_lrnow = frac * value_learning_rate
            f_approximator.optimizer.param_groups[0]["lr"] = f_approximator_lrnow

        for episode_step in tqdm(range(episode_steps_per_epoch)):
            # initalize per epoch variables
            action, log_prob = policy.act(observation)
            value = f_approximator(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            rewards_per_episode.append(reward)
            values_per_episode.append(value)

            # rtg and advantage will be calculated at trajectory end
            replay_buffer.buffer.append(
                BufferEntry(
                    observation=observation,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    rtg=np.nan,
                    advantage=np.nan,
                )
            )

            observation = next_observation

            epoch_ended = episode_step == (episode_steps_per_epoch - 1)

            # ensure proper bootstrapping for when time limit vs. terminal state reached (https://gymnasium.farama.org/environments/box2d/lunar_lander/#episode-termination)
            if truncated or terminated or epoch_ended:
                if truncated or (episode_step == epoch_ended):
                    final_value = f_approximator(observation)
                else:
                    final_value = 0
                replay_buffer.end_trajectory(episode_step, final_value)
                # per episode logging (note: currently does not log last episode on last epoch)
                if not epoch_ended:
                    total_reward, episode_length = (
                        info["episode"]["r"].item(),
                        info["episode"]["l"].item(),
                    )
                    observation, info = env.reset(seed=seed)

                    # log end of episode info
                    total_episodes += 1
                    rew_running_mean_buffer.append(total_reward)
                    wandb.log(
                        {
                            "episode_length": episode_length,
                            "unnormalized_advantages": wandb.Histogram(
                                [
                                    entry.advantage
                                    for entry in replay_buffer.buffer[
                                        episode_step : episode_step + episode_length
                                    ]
                                ]
                            ),
                            "rewards": wandb.Histogram(rewards_per_episode),
                            "values": wandb.Histogram(values_per_episode),
                            "reward_per_episode": total_reward,
                            "reward_per_episode_running_mean": (
                                sum(rew_running_mean_buffer)
                                / len(rew_running_mean_buffer)
                                if rew_running_mean_buffer
                                else 0.0
                            ),
                        }
                    )

                    # reset per episode variables
                    rewards_per_episode = []
                    values_per_episode = []
                    terminated = truncated = False

        observations, actions, rewards, values, log_probs, rtgs, advantages = (
            replay_buffer.get_trajectories()
        )
        old_policy_params = torch.cat([param.view(-1) for param in policy.parameters()])
        old_value_params = torch.cat(
            [param.view(-1) for param in f_approximator.parameters()]
        )

        # optimize for one epoch TODO use experiences more efficiently
        policy.optimizer.zero_grad()
        policy_loss, policy_metadata = policy.loss(
            observations, actions, advantages, log_probs
        )
        policy_loss.backward()
        policy.optimizer.step()

        for _ in range(value_steps_per_epoch):
            f_approximator.optimizer.zero_grad()
            value_loss = f_approximator.loss(observations, rtgs)
            value_loss.backward()
            f_approximator.optimizer.step()

        # per epoch metrics
        new_policy_params = torch.cat([param.view(-1) for param in policy.parameters()])
        new_value_params = torch.cat(
            [param.view(-1) for param in f_approximator.parameters()]
        )
        policy_step_MSE = ((new_policy_params - old_policy_params) ** 2).mean().item()
        value_step_MSE = ((new_value_params - old_value_params) ** 2).mean().item()

        policy_gradient_norms = [p.grad.norm().item() for p in policy.parameters()]
        value_gradient_norms = [
            p.grad.norm().item() for p in f_approximator.parameters()
        ]

        wandb.log(
            {
                "epoch": epoch,
                "total_episodes": total_episodes,
                "policy_entropy": policy_metadata[
                    "ent"
                ],  # used for checking how deterministic the policy is
                "approx_KL_divergence": policy_metadata[
                    "kl"
                ],  # used for checking how fast policy changing
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "policy_gradient_norms": max(
                    policy_gradient_norms
                ),  # used for checking for gradient exploding/vanishing
                "value_gradient_norms": max(value_gradient_norms),
                "policy_step_MSE": policy_step_MSE,  # used for lr tuning
                "value_step_MSE": value_step_MSE,
            }
        )

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == total_epochs:
            # TODO add hyperparameters
            save_checkpoint(
                policy.network,
                f_approximator.network,
                policy.optimizer,
                f_approximator.optimizer,
                epoch,
                hyperparams,
                CHECKPOINT_DIRECTORY + RUN_NAME,
                run.name,
            )

    env.close()
    return policy, run.name


if __name__ == "__main__":
    policy, policy_name = train()
    torch.save(policy, f"vpg_{policy_name}.pth")
