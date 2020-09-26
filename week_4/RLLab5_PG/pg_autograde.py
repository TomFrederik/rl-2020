import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.softmax(out, dim=1)
        return out
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        bs = obs.size(0)
        all_probs = self.forward(obs)
        action_probs = all_probs[range(bs), actions[:, 0]]
        return action_probs.view(bs, 1)
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        obs = obs.view(1, -1)
        action_probs = self.forward(obs)[0]
        print(self.forward(obs).shape)
        print(action_probs.shape)
        action = torch.multinomial(action_probs, 1).item()
        return action
        
        

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.softmax(out, dim=1)
        return out
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        bs = obs.size(0)
        all_probs = self.forward(obs)
        action_probs = all_probs[range(bs), actions[:, 0]]
        return action_probs.view(bs, 1)
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        obs = obs.view(1, -1) # ensure that first dim is 1
        action_probs = self.forward(obs)[0]
        action = torch.multinomial(action_probs, 1).item()
        return action
        
        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    states.append(torch.tensor(env.reset()).float())
    dones.append(False)
    while not dones[-1]:
        actions.append(policy.sample_action(states[-1]))
        obs, rew, done, _ = env.step(actions[-1])
        obs = torch.tensor(obs).float()
        if not done: 
            states.append(obs)
        rewards.append(rew)
        dones.append(done)
    # remove the first False in dones to make it the same length as the other lists
    dones = dones[1:]
    states = torch.tensor(np.stack(states, axis=0))
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards).unsqueeze(1)
    dones = torch.tensor(dones).unsqueeze(1)
    
    return states, actions, rewards, dones

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    # YOUR CODE HERE
    states, actions, rewards, dones = episode
    episode_len = rewards.size(0)
    G = torch.sum(rewards.squeeze() * discount_factor ** torch.arange(episode_len))
    action_probs = policy.get_probs(states, actions)
    loss = - action_probs.log().sum() * G
    
    return loss

# YOUR CODE HERE
#raise NotImplementedError

def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        
        # YOUR CODE HERE
        # don't apply gradient to sampling of episode
        with torch.no_grad():
            episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations
