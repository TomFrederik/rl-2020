import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

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
        
        # number of actions
        nA = self.Q.shape[1]
        
        # take random action with prob epsilon, greedy otherwise
        x = np.random.rand(1)
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            print(i)
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]
        
        # take random action with prob epsilon, greedy otherwise
        x = np.random.rand(1)
        print(x)
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        print(action)
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]
        
        # take random action with prob epsilon, greedy otherwise
        x = np.random.random_sample()
        print(x)
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        print(action)
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]
        print(nA)
        # take random action with prob epsilon, greedy otherwise
        x = np.random.random_sample()
        print(x)
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        print(action)
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        x = np.random.random_sample()
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
            print(action)
        ## ##
        
        print(action)
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        x = np.random.random_sample()
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
            print(action)
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        x = np.random.random_sample()
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
            print(self.Q[obs,:])
            print(action)
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        
        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
                print(action)

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            print(rew)
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            print(rew)
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            print(rew)
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            print(Q[last_obs, last_action])
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
            print(Q[last_obs, last_action])
            raise ValueError
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        x = np.random.random_sample()
        if x <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            print(Q[last_obs, last_action])
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
            print(Q[last_obs, last_action])
            raise ValueError
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            print(Q[last_obs, last_action])
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
            print(Q[last_obs, last_action])
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # update Q
            Q[last_obs, action] += 1/alpha * ( rew + discount_factor * np.max(Q[obs,:]) - Q[last_obs, action] )
            
        ## ##
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs
        latest_action = 0
        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            
            last_action = action 
            print('1:',last_action)
            action = policy.sample_action(obs)
            print('2:',last_action)
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        print(Q)
        print(policy.Q)
        raise ValueError
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            
            last_action = action 
            print('1:',last_action)
            action = policy.sample_action(obs)
            print(last_action)
            # update Q
            Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * Q[obs,action] - Q[last_obs, last_action] ) 
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action] ) 
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action] ) 
        
            print(policy.Q)
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * ( rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action] ) 
        
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',policy.Q - old_Q[policy.Q-old_Q!=0])
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(old_Q)
            print(policy.Q)
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',policy.Q - old_Q[policy.Q-old_Q!=0])
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(old_Q)
            print(policy.Q)
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',policy.Q - old_Q[(policy.Q-old_Q)!=0])
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(old_Q)
            print(policy.Q)
            # update Q
            print(policy.Q.shape)
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print(policy.Q.shape)
            print('Difference:',policy.Q - old_Q[(policy.Q-old_Q)!=0])
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(old_Q)
            print(policy.Q)
            # update Q
            print(policy.Q.shape)
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print(policy.Q.shape)
            print(old_Q.shape)
            print('Difference:',policy.Q - old_Q[(policy.Q-old_Q)!=0])
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(old_Q)
            print(policy.Q)
            # update Q
            print(policy.Q.shape)
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print(policy.Q.shape)
            print(old_Q.shape)
            print(old_Q - policy.Q)
            print('Difference:',policy.Q - old_Q[(policy.Q-old_Q)!=0])
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',np.sum(policy.Q - old_Q))
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action])
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',np.sum(policy.Q - old_Q))
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]))
            # update Q
            policy.Q[last_obs, last_action] += 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',np.sum(policy.Q - old_Q))
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            print(1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]))
            # update Q
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',np.sum(policy.Q - old_Q))
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            policy.Q = np.ones_like(policy.Q)
            print(policy.Q)
            raise ValueError
            print(1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]))
            # update Q
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',np.sum(policy.Q - old_Q))
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            old_Q = policy.Q
            policy.Q = np.ones_like(policy.Q)
            print(policy.Q)
            print(last_obs, last_action)
            raise ValueError
            print(1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]))
            # update Q
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',np.sum(policy.Q - old_Q))
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            print(policy.Q[last_obs, last_action])
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print(policy.Q[last_obs, last_action])
            raise ValueError
            print('Difference:',np.sum(policy.Q - old_Q))
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            print(policy.Q[last_obs, last_action])
            old_Q = policy.Q
            print(old_Q)
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print(policy.Q[last_obs, last_action])
            print(old_Q)
            print('Difference:',np.sum(policy.Q - old_Q))
            raise ValueError

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            print(policy.Q[last_obs, last_action])
            old_Q = policy.Q.copy()
            print(old_Q)
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print(policy.Q[last_obs, last_action])
            print(old_Q)
            print('Difference:',np.sum(policy.Q - old_Q))
            raise ValueError

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            print('Difference:',np.sum(policy.Q - old_Q))

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        last_obs = obs

        # simulate trajectory
        while not done:
            
            # take step
            obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action
            last_action = action 
            action = policy.sample_action(obs)
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[last_obs, last_action] = policy.Q[last_obs, last_action] + 1/alpha * (rew + discount_factor * policy.Q[obs,action] - policy.Q[last_obs, last_action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        if random.random() <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # update Q
            Q[obs, action] += 1/alpha * ( rew + discount_factor * np.max(Q[new_obs,:]) - Q[obs, action])
            
            obs = new_obs
            
        ## ##
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        
        ###
        self.ratio = 0
        self.n = 0
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        ## MY CODE ##
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        self.n += 1
        if random.random() <= self.epsilon: 
            action = np.random.randint(0,nA)
            self.ration += 1/self.n * (1-self.ratio)
            print('self.ratio = ', self.ratio)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        
        ###
        self.ratio = 0
        self.n = 0
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        ## MY CODE ##
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        self.n += 1
        if random.random() <= self.epsilon: 
            action = np.random.randint(0,nA)
            self.ratio += 1/self.n * (1-self.ratio)
            print('self.ratio = ', self.ratio)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        
        ###
        self.ratio = 0
        self.n = 0
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        ## MY CODE ##
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        self.n += 1
        if random.random() <= self.epsilon: 
            action = np.random.randint(0,nA)
            self.ratio += 1/self.n * (self.epsilon-self.ratio)
            print('self.ratio = ', self.ratio)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

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
        
        # number of actions
        nA = self.Q.shape[1]

        # take random action with prob epsilon, greedy otherwise
        if random.random() <= self.epsilon: 
            action = np.random.randint(0,nA)
        else: 
            action = np.argmax(self.Q[obs,:])
        ## ##
        
        return action

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##
    plt.ion()
    graph = plt.imshow(Q)[0]
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
            graph.set_data(policy.Q)
            graph.autoscale()
            plt.draw()
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##
    plt.ion()
    graph = plt.imshow(Q)
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
            graph.set_data(policy.Q)
            graph.autoscale()
            plt.draw()
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##
    plt.ion()
    graph = plt.imshow(Q)
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
            graph.set_data(policy.Q)
            graph.autoscale()
            plt.draw()
            plt.pause(0.01)
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##
    
    figure = plt.figure()
    graph = figure.imshow(Q)
    plt.ion()
    figure.show()
    figure.canvas.draw()
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
            graph.set_data(policy.Q)
            graph.autoscale()
            plt.draw()
            plt.pause(0.01)
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##
    
    figure = plt.figure()
    graph = plt.imshow(Q)
    plt.ion()
    figure.show()
    figure.canvas.draw()
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
            graph.set_data(policy.Q)
            graph.autoscale()
            plt.draw()
            plt.pause(0.01)
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##
    
    figure = plt.figure()
    graph = plt.imshow(Q)
    plt.ion()
    figure.show()
    figure.canvas.draw()
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
            graph.clear()
            graph.set_data(policy.Q)
            graph.autoscale()
            figure.canvas.draw()
            plt.pause(0.01)
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##
    
    figure = plt.figure()
    graph = plt.imshow(Q)
    plt.ion()
    figure.show()
    figure.canvas.draw()
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs
            
            graph.set_data(policy.Q)
            graph.autoscale()
            figure.canvas.draw()
            plt.pause(0.01)
            
        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            print('1: action is {}, new_action is{}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            print('2: action is {}, new_action is{}'.format(action, new_action))
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            print('1: action is {}, new_action is {}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            print('2: action is {}, new_action is {}'.format(action, new_action))
            
            # update Q
            #old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            #print('Difference:',np.sum(policy.Q - old_Q))
            
            action = new_action
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            print('1: action is {}, new_action is {}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            print('2: action is {}, new_action is {}'.format(action, new_action))
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0!')
            
            action = new_action
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            #print('1: action is {}, new_action is {}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            #print('2: action is {}, new_action is {}'.format(action, new_action))
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0!')
            
            action = new_action
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            #print('1: action is {}, new_action is {}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            #print('2: action is {}, new_action is {}'.format(action, new_action))
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0 in iteration {}!'.format(i))
            
            action = new_action
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            #print('1: action is {}, new_action is {}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            #print('2: action is {}, new_action is {}'.format(action, new_action))
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0 in iteration {}!'.format(i))
            
            action = new_action
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            #print('1: action is {}, new_action is {}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            #print('2: action is {}, new_action is {}'.format(action, new_action))
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0 in iteration {}!'.format(i))
            
            action = new_action
            print(type(new_obs))
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)
        new_action = 0
        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            #print('1: action is {}, new_action is {}'.format(action, new_action))
            new_action = policy.sample_action(new_obs)
            #print('2: action is {}, new_action is {}'.format(action, new_action))
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0 in iteration {}!'.format(i))
            
            action = new_action.copy()
            #print(type(new_obs))
            obs = new_obs.copy()

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0 in iteration {} of episode {}!'.format(i, i_epsiode))
            
            action = new_action
            #print(type(new_obs))
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + 1/alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            diff = np.sum(policy.Q - old_Q)
            if diff == 0:
                raise ValueError('difference between old and new Q is 0 in iteration {} of episode {}!'.format(i, i_episode))
            
            action = new_action
            #print(type(new_obs))
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    ##

    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            
            # take step
            new_obs, rew, done, _ = env.step(action)
            R += discount_factor**i * rew
            i += 1
            
            # sample new action 
            new_action = policy.sample_action(new_obs)
            
            # update Q
            old_Q = policy.Q.copy()
            policy.Q[obs,action] = policy.Q[obs,action] + alpha * (rew + discount_factor * policy.Q[new_obs,new_action] - policy.Q[obs,action]) 
            
            # update action and state
            action = new_action
            obs = new_obs

        ## ##
        
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            # take step
            new_obs, rew, done, _ = env.step(action)
            i += 1
            R += discount_factor**i * rew
            
            # update Q
            Q[obs, action] += alpha * ( rew + discount_factor * np.max(Q[new_obs,:]) - Q[obs, action])
            
            obs = new_obs
            
        ## ##
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            # take step
            new_obs, rew, done, _ = env.step(action)
            R += discount_factor**i * rew
            i += 1
            
            # update Q
            Q[obs, action] += alpha * ( rew + discount_factor * np.max(Q[new_obs,:]) - Q[obs, action])
            
            # update state
            obs = new_obs
            
        ## ##
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        ## MY CODE ##
        
        # init
        obs = env.reset()
        done = False
        action = policy.sample_action(obs)

        # simulate trajectory
        while not done:
            # take step
            new_obs, rew, done, _ = env.step(action)
            R += discount_factor**i * rew
            i += 1
            
            # update Q
            Q[obs, action] += alpha * ( rew + discount_factor * np.max(Q[new_obs,:]) - Q[obs, action])
            
            # update state and action
            obs = new_obs
            action = policy.sample_action(obs)
            
        ## ##
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)
