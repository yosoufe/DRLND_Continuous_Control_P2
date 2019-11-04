import torch
from torch import nn
import torch.optim as optim
from tools import ReplayBuffer, OUNoise

import numpy as np


class PPO_Agent:
    """
    DDPG Algorithm
    """

    def __init__(self,
                 state_size,
                 action_size,
                 actor_model,
                 critic_model,
                 device,
                 num_agents=1,
                 seed=0,
                 tau=1e-3,
                 batch_size=1024,
                 discount_factor=0.99,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3):
        """
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
            actor_learning_rate:
            critic_learning_rate:

        """
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.actor_local = actor_model(state_size, action_size, seed)
        self.actor_target = actor_model(state_size, action_size, seed)
        self.critic_local = critic_model(state_size, action_size, seed)
        self.critic_target = critic_model(state_size, action_size, seed)
        self.critic2_local = critic_model(state_size, action_size, seed+1)
        self.critic2_target = critic_model(state_size, action_size, seed+1)
        self.soft_update(1.0)
        self.batch_size = batch_size
        self.replayBuffer = ReplayBuffer(batch_size=batch_size, buffer_size=500000, seed=seed, device=device)
        self.num_agents = num_agents
        self.noise_process = OUNoise(action_size * num_agents, seed, max_sigma=0.5, min_sigma=0.01, decay_period=1000)
        self.discount_factor = discount_factor
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=actor_learning_rate)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=critic_learning_rate)
        self.critic2_opt = optim.Adam(self.critic2_local.parameters(), lr=critic_learning_rate)
        self.critic_criterion = nn.MSELoss()
        self.device = device
        self.st = 0
        for model in [self.actor_local,
                      self.actor_target,
                      self.critic_local,
                      self.critic_target,
                      self.critic2_local,
                      self.critic2_target]:
            model.to(device)

    def act(self, state, add_noise=True):
        """
        * Create actions using Actor Policy Network
        * Add noise to the actions and return it.

        Args:
            state: numpy array in shape of (num_agents, action_size).
            add_noise:

        Returns:
            actions: numpy arrays of size (num_agents, action_size)
        """
        state = torch.from_numpy(state).float().view(self.num_agents, self.state_size).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state)
            actions = actions.cpu().numpy()
            # print(actions[0])
        self.actor_local.train()
        if add_noise:
            actions += self.noise_process.sample().reshape(self.num_agents, self.action_size)
        return actions

    def step(self, states_ls, actions_ls, rewards_ls, next_states_ls, dones_ls):
        """
        For PPO what I need to calculate or create:
            reward to go or Advantage: tensor of size (trajectory length, number of agents, 1)
            states: tensor of size (trajectory length, number of agents, state_dim)
            actions: tensor of size (trajectory length, number of agents, action_dim)

        Two Actor's are needed:
            * The one that data has been gathered from
            * The one that is being updated
        Compute Advantage Estimate from the above estimate:
            Add the value from the last state to all rewards as a last point in trajectory (ignore it for now)
        Update Policy using PPO-Clip, needs:
            Advantage
            Actor is outputing probability of actions like a probability distribution
                torch.distributions.Normal could be used
            calculating rhe ratio of probabilities
        Update Value Function:

        Args:
            states_ls: list of np arrays of size (num_agents X state_dim)
            actions_ls: list of np arrays of size (num_agents X action_dim)
            rewards_ls: list of list of rewards
            next_states_ls:
            dones_ls:

        Returns:

        """
        # calculate the reward to go
        rewards = np.zeros()
        for reward in reversed(rewards_ls):
            # rewards_ls is list of number_agents X 1 floats
            pass

    def learn(self, states, actions, rewards, next_states, dones):
        """
        * sample a batch
        * set y from reward, Target Critic Network and Target Policy network
        * Calculate loss from y and Critic Network
        * Update the actor policy (would also update the critic by chain rule) using sampled policy gradient
        * soft update the target critic and target policy

        Args:
            actions:
            rewards:
            next_states:
            dones:

        Returns:
            None
        """
        # Update Critic
        next_actions = self.actor_target(next_states)
        # value = (self.critic_target(next_states, next_actions).detach() +
        #          self.critic2_target(next_states, next_actions).detach()) / 2.0
        value = torch.min(self.critic_target(next_states, next_actions).detach(),
                          self.critic2_target(next_states, next_actions).detach())
        y = rewards + self.discount_factor * value

        Q = self.critic_local(states, actions)
        critic_loss = self.critic_criterion(Q, y)

        Q2 = self.critic2_local(states, actions)
        critic2_loss = self.critic_criterion(Q2, y)

        # Update Actor
        action_predictions = self.actor_local(states)
        actor_loss = -self.critic_local(states, action_predictions).mean()

        # update networks
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # soft update
        self.soft_update(self.tau)
        self.st +=1

    def reset(self):
        self.noise_process.reset()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
