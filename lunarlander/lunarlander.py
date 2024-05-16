import gymnasium as gym
import tensorboard
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from collections import namedtuple

total_rewards = []
time_per_episode = []

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

ReplayEntry = namedtuple('ReplayEntry', ['observation', 'action', 'reward', 'next_observation', 'value', 'log_prob', 'rtg', 'advantage'])

class ReplayBuffer():
    """
    TODO
    """
    def __init__(self, batch_size, buffer_size=10000):
        self._buffer = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def append(self, entry):
        self._buffer.append(entry)
        # popleft if buffer > max size
        if(len(self._buffer) > self.buffer_size):
            self._buffer = self._buffer[1:]

    def sample(self):
        return np.random.choice(self._buffer, self.batch_size, replace=False)

class PolicyNN(nn.Module):
    """
    Fully connected NN w/ Epsilon greedy, only works for discrete actions spaces

    Outputs a probability for each action from the passed state which can then be sampled
    to choose the next action as well as log_prob for calculating gradients
    """
    def __init__(self, input_size=8, hidden_size=128, output_size=4, lr=1e-4):
        super(PolicyNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1) # Normalize probabilties to sum to one
        )
        self.epsilon = 0.05
        self.optimizer = Adam(self.parameters(), lr)

    def forward(self, x):
        x = torch.tensor(np.array(x))
        action_probs = self.network(x).squeeze()
        return Categorical(action_probs)

    def act(self, observation):
        action_dist = self.forward(observation)
        if np.random.rand() < self.epsilon:
            action = torch.tensor(np.random.randint(0, 3))
        else:
            action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def loss(self, observation, action, advantage, log_prob):
        action_dist = self.forward(observation, action)
        log_prob = action_dist.log_prob(action)
        return (log_prob * advantage)

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

    def forward(self, x) -> int:
        # TODO support batching
        x = torch.tensor(np.array(x))
        return self.network(x).item()

    def loss(self, observation, rtg):
        return (self.forward(observation) - rtg)**2


def train():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, _ = env.reset(seed=42)

    EPOCHS = 1
    EPISODES_PER_EPOCH = 10
    BATCH_SIZE = 32
    GAMMA = 0.99
    BUFFER_SIZE = 5000

    policy = PolicyNN()
    # function approximation must be used here since LunarLander's observation_space is continous
    f_approximator = ValueNN()
    replay_buffer = ReplayBuffer(BATCH_SIZE, BUFFER_SIZE)

    env = gym.make("LunarLander-v2", render_mode=None)
    TrajectoryStep = namedtuple('TrajectoryStep', ['observation', 'action', 'reward', 'next_observation', 'value', 'log_prob'])
    for _ in range(EPOCHS):
        times_per_episode = []
        total_rewards = []
        for episode in range(EPISODES_PER_EPOCH):
            observation, info = env.reset(seed=42)

            # initalize per episode variables
            time = 0
            total_reward = 0
            terminated = truncated = False

            # save to list first so that rewards-to-go and advantages can be calculated before appending to ReplayBuffer
            trajectory = []
            while not terminated or truncated:
                action, log_prob = policy.act(observation)
                value = f_approximator(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                trajectory.append(TrajectoryStep(observation=observation, action=action, reward=reward, next_observation=next_observation, value=value, log_prob=log_prob))

                observation = next_observation
                time += 1

            T = len(trajectory)
            # TODO
            if truncated:
                curr_rtg = f_approximator(observation)
            else:
                curr_rtg = 0

            # calculate rewards-to-go
            rewards_to_go = [0] * T
            for t in reversed(range(T)):
                curr_rtg = trajectory[t].reward + curr_rtg * GAMMA
                rewards_to_go[t] = curr_rtg
            
            # calculate advantages
            advantages = [(rtg - trajectory[t].value) for (t, rtg) in enumerate(rewards_to_go)]
            for t in range(T):
                replay_buffer.append(ReplayEntry(
                    observation=trajectory[t].observation,
                    action=trajectory[t].action,
                    reward=trajectory[t].reward,
                    next_observation=trajectory[t].next_observation,
                    value=trajectory[t].value,
                    log_prob=trajectory[t].log_prob,
                    rtg=rewards_to_go[t],
                    advantage=advantages[t]
                ))


            total_rewards.append(total_reward)
            times_per_episode.append(time)

        # optimize for one epoch
        
    env.close()
    return policy

def optimize(policy, f_approximator, _buffer):


if __name__ == "__main__":
    train()
