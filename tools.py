import torch
from torch import nn
from torch.nn import functional as F

from collections import deque, namedtuple
import random
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    Policy Model
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize and build the policy network.
        Args:
            state_size (int): The Dimension of states
            action_size (int): The Dimension of Action Space
            seed (int): seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """
        Forward path of the policy network.
        Args:
            state: tensor of states

        Returns:
            actions
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return F.tanh(x)


class Critic(nn.Module):
    """
    Critics Model
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize and build the critic network.
        Args:
            state_size (int): The Dimension of states
            action_size (int): The Dimension of Action Space
            seed (int): seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcState = nn.Linear(state_size, 128)
        self.fcAction = nn.Linear(action_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        Forward path of the critic network.
        Args:
            state: tensor of states
            action: tensor of actions

        Returns:
            qValue: Q(state,action)
        """
        xState = F.relu(self.fcState(state))
        xAction = F.relu(self.fcAction(action))
        x = F.relu(self.fc2((xState + xAction) / 2.0))
        x = F.relu(self.fc3(x))

        return x


class DDPG_Agent:
    """
    DDPG Algorithm
    """

    def __init__(self, state_size, action_size, num_agents=0, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_local = None
        self.critic_local = None
        self.num_agents = num_agents

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        return np.random.randn(self.num_agents, self.action_size)

    def reset(self):
        pass

    def learn(self):
        pass

    def soft_update(self):
        pass


class ReplayBuffer:
    """
    Replay Buffer

    Replay Buffer saves tuples of (s_t, a_t, r_t, s_(t+1), done) in circular buffer.
    Then it samples from them uniformly.
    """

    def __init__(self, batch_size, buffer_size=100000, seed=0):
        self.buffer = deque(maxlen=buffer_size)
        self.Experience = namedtuple("Experience",
                                     field_names=["state",
                                                  "action",
                                                  "reward",
                                                  "next_state",
                                                  "done"])
        self.seed = random.seed(seed)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_action, done):
        """
        Add an experience to the buffer.

        Args:
            state: 1D numpy array
            action: 1D numpy array
            reward: 1D numpy array
            next_action: 1D numpy array
            done (boolean): boolean

        Returns:

        """
        ex = self.Experience(state, action, reward, next_action, done)
        self.buffer.append(ex)

    def sample(self):
        ex = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in ex if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in ex if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in ex if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in ex if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in ex if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class PlotTool:
    """
    Example:
        from tools import PlotTool
        from time import sleep
        %matplotlib notebook
        plotter = PlotTool(number_agents=5)

        for i in range(3000):
            plotter.update_plot(np.random.randn(1,5))
            sleep(0.001)
    """
    def __init__(self, number_agents):
        self.number_agents = number_agents
        self.rewards = np.empty((number_agents, 1), dtype=np.float)
        self.average = np.empty((1, 1), dtype=np.float)
        self.initialized = False
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.autoscale()
        self.average_line = Line2D([0], [0], linestyle='-')
        self.average_line.set_label('Ave')
        self.average_line.set_color((0, 0, 0))
        self.ax.add_line(self.average_line)

        self.lines = [Line2D([0], [0], linestyle='--') for _ in range(number_agents)]
        for i, line in enumerate(self.lines):
            self.ax.add_line(line)
            line.set_color((random.random(), random.random(), random.random()))
            line.set_label('A{}'.format(i))
        assert len(self.lines) == number_agents

    def update_plot(self, newRewards):
        newRewards = newRewards.reshape(1, self.number_agents)
        if not self.initialized:
            self.rewards = newRewards
            self.average = np.mean(self.rewards, axis=1).reshape((1,1))
            self.initialized = True
        else:
            self.rewards = np.concatenate((self.rewards, newRewards), axis=0)
            self.average = np.concatenate((self.average, np.mean(newRewards).reshape((1,1)) ), axis=0)

            for li, line in enumerate(self.lines):
                line.set_xdata(np.arange(self.rewards.shape[0]))
                line.set_ydata(self.rewards[:, li])
            self.average_line.set_xdata(np.arange(self.average.shape[0]))
            self.average_line.set_ydata(self.average)

            self.ax.legend(bbox_to_anchor=(.99, 1), loc='upper left', ncol=1)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
