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
                Q_values = self.Q(torch.from_numpy(obs).float())
                action = torch.argmax(Q_values).item()
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    raise NotImplementedError
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    raise NotImplementedError

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = out[np.arange(out.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = out[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)
    print('max_qs.shape = ', max_qs)
    print('max_qs = ', max_qs)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    print(all_qs)
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)
    print('max_qs.shape = ', max_qs)
    print('max_qs = ', max_qs)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    print('max_qs.shape = ', max_qs)
    #print('max_qs = ', max_qs)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    print('max_qs.shape = ', max_qs)
    #print('max_qs = ', max_qs)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    print('max_qs.shape = ', max_qs)
    #print('max_qs = ', max_qs)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    print(target.shape)
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    print('max_qs.shape = ', max_qs)
    #print('max_qs = ', max_qs)
    
    # compute targets
    targets = rewards + discount_factor * max_qs
    
    print(targets.shape)
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    print('max_qs.shape = ', max_qs)
    #print('max_qs = ', max_qs)
    
    # compute targets
    print(rewards.shape)
    print(discount_factor)
    targets = rewards + discount_factor * max_qs
    
    print(targets.shape)
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    print('max_qs.shape = ', max_qs.shape)
    #print('max_qs = ', max_qs)
    
    # compute targets
    print(rewards.shape)
    print(discount_factor)
    targets = rewards + discount_factor * max_qs
    
    print(targets.shape)
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    print('max_qs.shape = ', max_qs.shape)
    #print('max_qs = ', max_qs)
    
    # compute targets
    print(rewards.shape)
    print(discount_factor)
    targets = rewards + discount_factor * max_qs[:,None]
    
    print(targets.shape)
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]-> need to do some broadcasting shenanigans
    
    # compute targets
    targets = rewards + discount_factor * max_qs[:,None]
    
    return targets[:,0]
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]-> need to do some broadcasting shenanigans
    
    # compute targets
    targets = rewards[:,0] + discount_factor * max_qs#[:,None]
    
    return targets#[:,0]
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]
    
    # compute targets
    targets = rewards[:,0] + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            # sample a batch from memory
            batch = memory.sample(batch_size)
            
            # zero grad
            optimizer.zero_grad()
            
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            # backwards pass
            loss.backward()
            
            # update Q_net
            optimizer.step()
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            # sample a batch from memory
            batch = memory.sample(batch_size)
            
            # zero grad
            optimizer.zero_grad()
            
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size), len(memory))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            print(action)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

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
            action = random.randint(0,1)
        else:
            with torch.no_grad():
                Q_values = self.Q(torch.from_numpy(obs).float())
                action = torch.argmax(Q_values).item()
        
        return action
        ## ##
        
        
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]
    
    # compute targets
    targets = rewards[:,0] + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            print(action)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            print(done)
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)
            print(done)
            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            # inc step counter
            steps += 1
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)

            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            # inc step counter
            steps += 1
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    print(q_vals)
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]
    
    # compute targets
    targets = rewards[:,0] + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    print(all_qs)
    print(actions)
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    print(q_vals)
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]
    
    # compute targets
    targets = rewards[:,0] + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    print(q_vals)
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]
    
    # compute targets
    targets = rewards[:,0] + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)

            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            # inc step counter
            steps += 1
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    ## MY CODE ##
    
    # perform a forward pass
    all_qs = Q(states) # should have shape batch_size x |A|
    #print('all_qs.shape = ', all_qs.shape)
    
    # get q values of actions that were actually taken
    q_vals = all_qs[np.arange(all_qs.shape[0]), actions[:,0]] # should have shape batch_size x 1
    #print('q_vals.shape = ', q_vals.shape)
    
    return q_vals
    ## ##
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    ## MY CODE ##
    
    # compute all q_vals
    all_qs = Q(next_states)
    #print('all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # if a state was done, then the q-values should be set to zero:
    all_qs = ~done * all_qs
    #print('masked all_qs.shape = ', all_qs.shape) # should have shape batch_size x |A|
    
    # get the maximum q_value for each next state
    max_qs = torch.max(all_qs, dim=1)[0] # only want value, not action
    #print('max_qs.shape = ', max_qs.shape) # is of shape [64]
    
    # compute targets
    targets = rewards[:,0] + discount_factor * max_qs
    
    return targets
    ## ##

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            ## MY CODE ##
            
            # get current epsilon
            eps = get_epsilon(global_steps)
            
            # set eps for policy
            policy.set_epsilon(eps)
            
            # get new action
            action = policy.sample_action(state)
            
            # get new transition
            next_state, reward, done, _ = env.step(action)

            # store transition in memory
            memory.push((state, action, reward, next_state, done))
            
            # set s <= s'
            state = next_state
            
            # sample a batch from memory
            batch = memory.sample(min(batch_size, len(memory)))
                        
            # compute batch loss
            loss = train(Q_net, memory, optimizer, batch_size, discount_factor)
            
            # inc step counter
            steps += 1
            
            ## ##
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations
