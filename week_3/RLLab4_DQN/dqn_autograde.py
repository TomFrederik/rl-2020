import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        ## My Code ##
        out = self.l1(x)
        out = nn.functional.relu(out)
        out = self.l2(x)
        
        return out
        ## ##
        

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        ## My Code ##
        print(x.shape)
        out = self.l1(x)
        out = nn.functional.relu(out)
        out = self.l2(x)
        
        return out
        ## ##
        

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        ## My Code ##
        
        out = self.l1(x.T)
        out = nn.functional.relu(out)
        out = self.l2(x)
        
        return out
        ## ##
        

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        ## My Code ##
        
        out = self.l1(x)
        out = nn.functional.relu(out)
        out = self.l2(x)
        
        return out
        ## ##
        

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        ## My Code ##
        print(x.shape)
        out = self.l1(x)
        print(out.shape)
        out = nn.functional.relu(out)
        out = self.l2(x)
        
        return out
        ## ##
        

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        ## My Code ##
        print(x.shape)
        out = self.l1(x)
        print(out.shape)
        out = nn.functional.relu(out)
        print(out.shape)
        out = self.l2(x)
        
        return out
        ## ##
        

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        ## My Code ##
        out = self.l1(x)
        out = nn.functional.relu(out)
        out = self.l2(out)
        
        return out
        ## ##
        

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        ## MY CODE ##
        
        # if at full capacity, remove oldest transition
        if self.capacity == self.len:
            del self.memory[0]
            
        # add new transition
        self.memory.append(transition)
        
        ## ##

    def sample(self, batch_size):
        ## MY CODE ##
        
        # get a sample from the memory
        sample = random.sample(self.memory, batch_size)
        
        return sample
        ## ##

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        ## MY CODE ##
        
        # if at full capacity, remove oldest transition
        if self.capacity == self.len():
            del self.memory[0]
            
        # add new transition
        self.memory.append(transition)
        
        ## ##

    def sample(self, batch_size):
        ## MY CODE ##
        
        # get a sample from the memory
        sample = random.sample(self.memory, batch_size)
        
        return sample
        ## ##

    def __len__(self):
        return len(self.memory)

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        ## MY CODE ##
        
        # if at full capacity, remove oldest transition
        if self.capacity == self.__len__():
            del self.memory[0]
            
        # add new transition
        self.memory.append(transition)
        
        ## ##

    def sample(self, batch_size):
        ## MY CODE ##
        
        # get a sample from the memory
        sample = random.sample(self.memory, batch_size)
        
        return sample
        ## ##

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    #$ MY CODE ##
    epsilon = 1 - (0.95/1000 * it)
    
    ## ##
    return epsilon

def get_epsilon(it):
    #$ MY CODE ##
    epsilon = min(0.05, 1 - (0.95/1000 * it))
    
    ## ##
    return epsilon

def get_epsilon(it):
    #$ MY CODE ##
    epsilon = max(0.05, 1 - (0.95/1000 * it))
    
    ## ##
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        
        if random.random() <= self.epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                Q_values = self.Q(obs)
                action = int(np.argmax(Q_values))
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        
        if random.random() <= self.epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.from_nump(obs))
                print(Q_values)
                action = int(np.argmax(Q_values))
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        
        if random.random() <= self.epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.from_numpy(obs))
                print(Q_values)
                action = int(np.argmax(Q_values))
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        
        if random.random() <= self.epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.from_numpy(obs).double())
                print(Q_values)
                action = int(np.argmax(Q_values))
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        
        if random.random() <= self.epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.from_numpy(obs).float())
                print(Q_values)
                action = int(np.argmax(Q_values))
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        
        if random.random() <= self.epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.from_numpy(obs).float())
                action = torch.argmax(Q_values)
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        
        if random.random() <= self.epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.from_numpy(obs).float())
                action = torch.argmax(Q_values).item()
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
