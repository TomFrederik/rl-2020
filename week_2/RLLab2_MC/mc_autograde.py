import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        
        ## MY CODE ##
        assert len(states) == len(actions), "There must be equally many states and actions"
        probs = []
        for state, action in zip(states, actions):
            # works only because the policy is deterministic
            if self.sample_action(state) == action:
                probs.append(1)
            else:
                probs.append(0)

        ## ##
        
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        ## MY CODE ##
        if state[0] >= 20:
            action = 0
        else:
            action = 1
        ## ##
        
        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    ## MY CODE ##
    states.append(env.reset())
    dones.append(False)
    while not dones[-1]:
        actions.append(policy.sample_action(states[-1]))
        obs, rew, done, info = env.step(actions[-1])
        if not done: 
            states.append(obs)
        rewards.append(rew)
        dones.append(done)
    ## ##
    return states, actions, rewards, dones

def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    ## MY CODE ##
    # first visit
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        # keeps track of first visit
        already_updated = defaultdict(bool)
        # keeps track of episode return
        episode_returns = defaultdict(float)
        
        for j in range(len(states))[::-1]:
            
            if not(states[j] in already_updated):
                already_updated[states[j]] =  True
                returns_count[states[j]] += 1
                episode_returns[states[j]] = 0
                V[states[j]] = 0

            episode_returns[states[j]] *= discount_factor
            episode_returns[states[j]] += rewards[j]

        # update value function
        
        for state in V.keys():
            V[state] += 1/returns_count[state] * (episode_returns[state] - V[state])
                
                
            
    
    ## ##
    
    return V
