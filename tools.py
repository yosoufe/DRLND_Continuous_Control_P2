import torch
from torch import nn
import torch.optim as optim
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
        self.tanh = nn.Tanh()

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

        return self.tanh(x)


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
        x = self.fc3(x)
        return x


class DDPG_Agent:
    """
    DDPG Algorithm
    """

    def __init__(self,
                 state_size,
                 action_size,
                 num_agents=1,
                 seed=0,
                 tau=1e-3,
                 batch_size=512,
                 discount_factor = 0.99,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3):
        """
        TODO:
        Initialize the 4 networks
        Copy 2 of them into the other two:
        * actor and actor_target
        * critic and critic_target
        init the replay buffer and the noise process

        Args:
            state_size:
            action_size:
            num_agents:
            seed:
            tau:
            batch_size:
            discount_factor:

        """
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.soft_update(1.0)
        self.batch_size = batch_size
        self.replayBuffer = ReplayBuffer(batch_size=batch_size, buffer_size=100000, seed=seed)
        self.num_agents = num_agents
        self.noise_process = OUNoise(action_size * num_agents, seed, theta=0.05)
        self.discount_factor = discount_factor
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=actor_learning_rate)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=critic_learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state, add_noise=True):
        """
        TODO:
        * Create actions using Actor Policy Network
        * Add noise to the actions and return it.

        Args:
            state: numpy array in shape of (num_agents, action_size)

        Returns:
            actions: numpy arrays of size (num_agents, action_size)
        """
        state = torch.from_numpy(state).float().view(self.num_agents, self.state_size).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise_process.sample().reshape(self.num_agents, self.action_size)
        return actions

    def step(self, state, action, reward, next_state, done):
        """
        TODO:
        * save sample in the replay buffer
        * if replay buffer is large enough
            * sample a batch
            * set y from reward, Target Critic Network and Target Policy network
            * Calculate loss from y and Critic Network
            * Update the actor policy (would also update the critic by chain rule) using sampled policy gradient
            * soft update the target critic and target policy

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:
            None
        """
        self.replayBuffer.push(state, action, reward, next_state, done)
        if len(self.replayBuffer) > self.batch_size:
            states, actions, rewards, next_states, dones = self.replayBuffer.sample()
            next_actions = self.actor_local(next_states)
            y = rewards + \
                self.discount_factor * self.critic_target( next_states, next_actions.detach() )
            Q = self.critic_local(states, actions)
            critic_loss = self.criterion(Q, y)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            action_predictions = self.actor_local(states)
            actor_loss = -self.critic_local(states,action_predictions).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.tau)

    def reset(self):
        self.noise_process.reset()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


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

    def push(self, state, action, reward, next_states, done):
        """
        Add an experience to the buffer.

        Args:
            state: 2D numpy array in size of number of agents by number of states
            action: 2D numpy array in size of number of agents by number of actions
            reward: list in size of number of agents
            next_states: same as state
            done (boolean): list in size of number of agents

        Returns:
            None
        """
        for i in range(state.shape[0]):
            ex = self.Experience(state[i, :], action[i, :], reward[i], next_states[i, :], done[i])
            self.buffer.append(ex)

    def sample(self):
        """
        sample from the buffer

        Returns:
            state: tensor in size (batch_size, number of states)
            action: tensor in size (batch_size, number of actions)
            reward: tensor in size (batch_size, 1)
            next_state: same as state
            done (boolean): tensor in size (batch_size, number of states)
        """
        ex = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in ex if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in ex if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in ex if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in ex if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in ex if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """

        Returns:
            number of experiences in the buffer
        """
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.uniform(-1.0, 1.0) for i in range(len(x))])
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
        self.fig = plt.figure(figsize=(8, 8))
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
            self.average = np.mean(self.rewards, axis=1).reshape((1, 1))
            self.initialized = True
        else:
            self.rewards = np.concatenate((self.rewards, newRewards), axis=0)
            self.average = np.concatenate((self.average, np.mean(newRewards).reshape((1, 1))), axis=0)

            for li, line in enumerate(self.lines):
                line.set_xdata(np.arange(self.rewards.shape[0]))
                line.set_ydata(self.rewards[:, li])
            self.average_line.set_xdata(np.arange(self.average.shape[0]))
            self.average_line.set_ydata(self.average)

            self.ax.legend(bbox_to_anchor=(.99, 1), loc='upper left', ncol=1)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
