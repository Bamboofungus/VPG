import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any
from collections import namedtuple
from torch.types import Number
from tqdm import tqdm

import wandb

if not torch.cuda.is_available():
    print("gpu not detected, falling back to cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def manual_policy(obs) -> int:
    """
    Naive policy that tries to orient lander legs toward ground and slowing near ground for comparison to
    learned policy
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = obs
    # calculate current downward vector
    if left_contact or right_contact:
        return 0
    elif vy < -1 and y > 0.1:
        print(vy)
        return 2
    else:
        if angle < -0.5:
            return 1
        elif angle > 0.5:    
            return 3
    # default
    return 0

BufferEntry = namedtuple('BufferEntry', ['observation', 'action', 'reward', 'value', 'log_prob', 'rtg', 'advantage'])

class TrajectoryBuffer():
    """
    Buffer for storing data from policy interacting with environment for use in training at end
    """
    def __init__(self, buffer_size, gamma):
        self.buffer = []
        self.buffer_size = buffer_size
        self.curr_trajectory_start = 0
        self.gamma = gamma

    def end_trajectory(self, trajectory_end, final_value=0):
        curr_rtg = final_value

        for t in reversed(range(self.curr_trajectory_start, trajectory_end + 1)):
            curr_rtg = self.buffer[t].reward + curr_rtg * self.gamma
            self.buffer[t] = BufferEntry(
                observation=self.buffer[t].observation,
                action=self.buffer[t].action,
                reward=self.buffer[t].reward,
                value=self.buffer[t].value,
                log_prob=self.buffer[t].log_prob,
                rtg=curr_rtg,
                advantage=curr_rtg - self.buffer[t].value
            )

        self.curr_trajectory_start = trajectory_end + 1


    def get_trajectories(self):
        # assert len(self._buffer) > self.batch_size, "TrajectoryBuffer not populated enough to sample from"
        # indices = np.random.choice(len(self._buffer), self.batch_size, replace=False)
        # batch = [self._buffer[idx] for idx in indices]

        # separate observations, actions, etc. into different tensors
        observations, actions, rewards, values, log_probs, rtgs, advantages = zip(*self.buffer)
        observations = torch.tensor(np.array(observations), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(device)
        log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(device)
        rtgs = torch.tensor(np.array(rtgs), dtype=torch.float32).to(device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(device)

        self.buffer = []
        self.curr_trajectory_start = 0

        return observations, actions, rewards, values, log_probs, rtgs, advantages


class PolicyNN(nn.Module):
    """
    Fully connected NN w/ Epsilon greedy, only works for discrete actions spaces

    Outputs a probability for each action from the passed state which can then be sampled
    to choose the next action as well as log_prob for calculating gradients
    """
    def __init__(self, input_size=8, hidden_size=128, output_size=4, lr=1e-4, epsilon=0.05):
        super(PolicyNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1) # Normalize probabilties to sum to one
        )
        self.epsilon = epsilon
        self.optimizer = Adam(self.parameters(), lr)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(device)
        action_probs = self.network(x).squeeze()
        return Categorical(action_probs)

    def act(self, observation) -> Tuple[Number, float]:
        with torch.no_grad():
            action_dist = self.forward(observation)
            if np.random.rand() < self.epsilon:
                action = torch.tensor(np.random.randint(0, 3)).to(device)
            else:
                action = action_dist.sample()
            log_prob = action_dist.log_prob(action).item()
        return action.item(), log_prob

    def loss(self, observations, actions, advantages, old_log_probs):
        action_dists = self.forward(observations)
        log_probs = action_dists.log_prob(actions)
        # important: negate this loss since it's meant for gradient ascent
        return - (log_probs * advantages).mean()

class ValueNN(nn.Module):
    """
    Outputs the estimated value of a passed in state which can then be used in the advantage term
    """
    def __init__(self, input_size=8, hidden_size=128, output_size=1, lr=1e-4):
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
        # call network directly here since forward doesn't support batching
        return ((self.network(observations) - rtgs)**2).mean()


def train():
    env = gym.make("LunarLander-v2", render_mode=None)
    observation, _ = env.reset(seed=42)

    EPOCHS = 50
    EPISODE_STEPS_PER_EPOCH = 4000
    # BATCH_SIZE = 256
    GAMMA = 0.99
    POLICY_LEARNING_RATE = 1e-3
    VALUE_LEARNING_RATE = 3e-3
    EPSILON = 0.05
    LOGGING = False
    VALUE_STEPS_PER_EPOCH = 10
    
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="lunarlander_vpg",
        # Track hyperparameters and run metadata 
        mode= "online" if LOGGING else "disabled", 
        config={ 
            "epochs": EPOCHS,
            # "batch_size": BATCH_SIZE,
            "policy_learning_rate": POLICY_LEARNING_RATE,
            "value_learning_rate": VALUE_LEARNING_RATE,
            "epsilon": EPSILON,
        },
    )

    policy = PolicyNN(lr=POLICY_LEARNING_RATE, epsilon=EPSILON).to(device)
    # function approximation must be used here since LunarLander's observation_space is continous
    f_approximator = ValueNN(lr=VALUE_LEARNING_RATE).to(device)
    replay_buffer = TrajectoryBuffer(EPISODE_STEPS_PER_EPOCH, GAMMA)


    # initialize per episode variables
    rewards_per_episode = []
    values_per_episode = []
    episode_length = 0
    total_reward = 0
    terminated = truncated = False

    for epoch in range(EPOCHS):
        print(f"Running epoch #{epoch}")

        for episode_step in tqdm(range(EPISODE_STEPS_PER_EPOCH)):
            # initalize per epoch variables
            terminated_episodes = 0
            action, log_prob = policy.act(observation)
            value = f_approximator(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            rewards_per_episode.append(reward)
            values_per_episode.append(value)

            # rtg and advantage will be calculated at trajectory end
            replay_buffer.buffer.append(
                BufferEntry(observation=observation, action=action, reward=reward, value=value, log_prob=log_prob, rtg=np.nan, advantage=np.nan)
            )

            observation = next_observation
            episode_length += 1

            # ensure bootstrapping correct for when time limit vs. terminal state reached (https://gymnasium.farama.org/environments/box2d/lunar_lander/#episode-termination)
            epoch_ended = episode_step == (EPISODE_STEPS_PER_EPOCH - 1)
            if truncated or terminated or epoch_ended:
                if truncated or (episode_step == epoch_ended):
                    final_value = f_approximator(observation)
                else:
                    final_value = 0
                    terminated_episodes += 1
                replay_buffer.end_trajectory(episode_step, final_value)
                # carry over episodes between epochs
                if not epoch_ended and not (epoch == EPOCHS - 1):
                    observation, info = env.reset(seed=42)

                    # log end of episode info
                    wandb.log({
                        "episode_length": episode_length,
                        "advantages": wandb.Histogram([entry.advantage for entry in replay_buffer.buffer[episode_step:episode_step + episode_length]]),
                        "rewards": wandb.Histogram(rewards_per_episode),
                        "values": wandb.Histogram(values_per_episode),
                        "reward_per_episode": total_reward
                    })

                    # reset per episode variables
                    rewards_per_episode = []
                    values_per_episode = []
                    episode_length = 0
                    total_reward = 0
                    terminated = truncated = False

        observations, actions, rewards, values, log_probs, rtgs, advantages = replay_buffer.get_trajectories()
        old_policy_params = torch.cat([param.view(-1) for param in policy.parameters()])
        old_value_params = torch.cat([param.view(-1) for param in f_approximator.parameters()])

        # optimize for one epoch TODO use experiences more efficiently
        policy.optimizer.zero_grad()
        policy_loss = policy.loss(observations, actions, advantages, log_probs)
        policy_loss.backward()
        policy.optimizer.step()

        for _ in range(VALUE_STEPS_PER_EPOCH):
            f_approximator.optimizer.zero_grad()
            value_loss = f_approximator.loss(observations, rtgs)
            value_loss.backward()
            f_approximator.optimizer.step()
        
        # per epoch metrics
        new_policy_params = torch.cat([param.view(-1) for param in policy.parameters()])
        new_value_params = torch.cat([param.view(-1) for param in f_approximator.parameters()])
        policy_step_MSE = ((new_policy_params - old_policy_params) ** 2).mean().item()
        value_step_MSE = ((new_value_params - old_value_params) ** 2).mean().item()

        policy_gradient_norms = [p.grad.norm().item() for p in policy.parameters()]
        value_gradient_norms = [p.grad.norm().item() for p in f_approximator.parameters()]

        wandb.log({
            "epoch": epoch,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "policy_gradient_norms": max(policy_gradient_norms), # used for checking for gradient exploding/vanishing
            "value_gradient_norms": max(value_gradient_norms),
            "policy_step_MSE": policy_step_MSE, # used for lr tuning
            "value_step_MSE": value_step_MSE
        })
    env.close()
    return policy

if __name__ == "__main__":
    train()
