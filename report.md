# Report

## Introduction:

The environment and the goal is introduced in main readme file [here](readme.md). 

## Learning Algorithm:

Deep Deterministic Policy Gradients (**DDPG**) is used to train an RL agent to score higher in the environment.

### Deep Deterministic Policy Gradients (**DDPG**):

DDPG is introduced in the paper 
[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).

Quick Facts about DDPG:
* It is an off-policy algorithm.
* It can be used only for environments with continuous action spaces.
* It can be considered as deep Q-learning for continuous action spaces while it is introduced as Actor Critic method.

DDPG is an algorithm that learns both the policy and value function together. It contains of 4 neural networks as 
following:

* <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta^{Q}"/> : Q network or what 
I called `critic_local` in my implementation:
    * Input: states and actions
    * Output: value of the state and action
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta^{\mu}"/> : Deterministic policy 
function or what I called `actor_local` in my implementation:
    * Input: states
    * Output: best believed actions
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta^{Q'}"/> : target Q network or what I 
called `critic_target` in my implementation which is similar to Q network.
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta^{\mu'}"/> : target policy function or 
what I called `actor_target` in my implementation which is similar to policy function.

In this algorithm, actor network is used to approximate the optimal policy deterministically. It output the best
believed action for a given state. Actor is basically predicts 
<img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;\operatorname*{argmax}_{a}Q(s,a)"/> which 
is the best action. Critic learns to evaluate the optimal action-value function by using the actor's best believed 
action. Which is an approximate maximizer to calculate new target value for training the action value function.

### Pseudocode 
![Pseudocode](imgs_vids/pseudocode.png "Pseudocode taken from the paper \"Continuous control with deep reinforcement learning\"")

### Models:
Neural Network Models are defined in [models.py](models.py)

#### Actor or Policy:
Actor or Policy model consists of:
* Batch normalization layer at the state inputs. The state values can have very large and small values in different 
dimensions which batch norm can speed up the learning. 
* Multiple linear layers
* tanh as last activation because we know the maximum and minimum values for actions are +1 and -1. `Tanh` outputs in 
the same range which makes the learning faster. Be careful that `relu` should not be used at last layer because `relu` outputs
only zero and positive values. I did this mistake and spent some time to find the bug.

The input of model is the states and outputs the best believed actions.
```python
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):
        x = state
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))
```

#### Critic or Q Function:
Critic or Q Function model consists of:
* Multiple linear layers
* And no (or linear) activation at last layer.

The input of model is the states and action and outputs the 
<img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s,a)"/>.
```python
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcState = nn.Linear(state_size, 32)
        self.fcAction = nn.Linear(action_size, 16)
        self.fc2 = nn.Linear(32+16, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, state, action):
        xState = F.relu(self.fcState(state))
        xAction = F.relu(self.fcAction(action))
        x = torch.cat((xState, xAction), 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
```

#### Networks Update:

##### Critic
First we calculate the target value for Q function using target critic network 
<img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;Q'"/> as follow

<img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;y_i=r_i+\gamma%20Q'(s_{i+1},\mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})"/>

Then we calculate the loss 

<img align="middle" src="https://latex.codecogs.com/svg.latex?\Large&space;Loss=\dfrac{1}{N}\operatorname*{\Sigma}_{i}{(y_i-Q(s_i,a_i|\theta^Q))^2}"/>

##### Actor
For actor, the objective is to maximize the expected return

<img src="https://latex.codecogs.com/svg.latex?\Large&space;J(\theta)=\mathbb{E}\[Q(s,a)|_{s=s_{t},a_t=\mu(s_t)}\]"/>.

We need to calculate the objective's gradient using chain rule

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\Delta_{\theta^{\mu}}\approx\Delta_a%20Q(s,a)\Delta_{\theta^{\mu}}\mu(s|\theta^\mu)"/>.


#### Target Networks Update or Soft Update
It is similar to deep Q-learning method:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta^{Q'}\leftarrow\tau\theta^{Q}+(1-\tau)\theta^{Q'}"/>

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta^{\mu'}\leftarrow\tau\theta^{\mu}+(1-\tau)\theta^{\mu'}"/>

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\tau<<1"/>.
```python
def soft_update(self, tau):
    for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
```

#### Experience Replay or Replay Buffer:
Learning on single sample is usually ending up having a very high variance. Therefore it is essential to 
have a buffer to save the experiments. Then on each step random samples are drawn from the buffer and 
the network is trained on them. This is like we are replaying the old experience. This is very similar to 
deep Q-learning.

#### Exploration
We are using [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
added to actions. We are decreasing noise further more in learning.

### Hyperparameters and Tuning:
I found DDPG very sensitive to hyper parameters. If the parameters were not chosen correctly
the output of the actor saturates to the extremes very early in the training and comes out of that 
local optima rarely. The corect hyperparameters are also depending on the networks and its complexity.
Here are the chosen parameters:

```python
agent = DDPG_Agent(state_size=state_size, 
                   action_size=action_size, 
                   actor_model=Actor,
                   critic_model=Critic,
                   device=device,
                   num_agents= num_agents, 
                   seed=1,
                   tau=3e-1,
                   batch_size=2048,
                   discount_factor = 0.99,
                   actor_learning_rate=1e-4,
                   critic_learning_rate=1e-3)
```

I used two critic networks with the same architecture but different initial parameters
and used the average of their output as the target value for critics network. This gave a bit
more stable convergence.

## Result:

In the following image, the top graph is showing the rewards of each agent and the second 
graph is showing the average over all agents. I set the criteria to consider the environment as 
solve to average rewards of **35** and this is achieved in **140 epochs**

<img src="imgs_vids/Rewards.png" alt="drawing" height="900"/>

Here are some videos of my trained agent:
* with live cumulative rewards: https://youtu.be/ZCwCc8ClnbA
* without cumulative rewards: https://youtu.be/K_hSIYBih-c

And the gif version of the videos, You may click on them to go to High Quality Version on 
Youtube, Or may check the `imgs_vids` folder of the repo:

[![Trained Agent](imgs_vids/ddpg_with_rewards.gif "My Trained Agent, Click for Youtube Video")](https://youtu.be/ZCwCc8ClnbA)
[![Trained Agent](imgs_vids/ddpg_full_screen.gif "My Trained Agent, Click for Youtube Video")](https://youtu.be/K_hSIYBih-c)

## Future Ideas:
* Using different algorithm like PPO, A2C, A3C or TD3.
* Using different network models like recurrent networks.

# Refereces
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* [Deep Deterministic Policy Gradient, OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
* [Deep Deterministic Policy Gradients Explained, Medium Article](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)