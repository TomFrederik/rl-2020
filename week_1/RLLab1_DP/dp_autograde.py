import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    
    ## MY CODE ##
    done = False
    ctr = 0
    while not done:
        ctr += 1
        # compute the next iteration of V
        V_new = np.zeros_like(V)
        
        for s in range(env.nS):
            
            # update state s
            for a in range(env.nA):
                
                for transition in env.P[s][a]:    
                    
                    # sum over all transition resulting from taking action a in state s
                    prob = transition[0]
                    next_state = transition[1]
                    rew = transition[2]
                    V_new[s] += policy[s,a]* prob * (rew + discount_factor * V[next_state])

            
        # check if stopping criterion is reached, i.e. that the largest update is smaller than theta
        max_update = np.max(np.abs(V_new - V))
        if max_update < theta:
            done = True
        
        # save new V
        V = V_new
        
    ## END MY CODE ###
    
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    ## MY CODE ##
    
    # initial evaluation
    V = policy_eval_v(policy, env, discount_factor)
    
    
    # update until done
    done = False
    
    while not done:
        
        # improve policy by acting greedily w.r.t. our current V
        # for this we need to compute the argmax over actions
        # new_policy(s) = argmax_{a} \sum_{r,s'} p(s',r|s,a) ( r + \gamma V(s') )
        new_policy = np.zeros_like(policy)
        
        for s in range(env.nS):
            
            highest_value = -np.inf
            
            for a in range(env.nA):    
                # compute value of taking action a and following policy afterwards
                value = 0                
                for transition in env.P[s][a]:
                    prob = transition[0]
                    next_state = transition[1]
                    rew = transition[2]
                    value += prob * (rew + discount_factor * V[next_state])
                
                if value > highest_value:
                    highest_value = value
                    best_action = a
            
            # set new policy to best action
            new_policy[s] = np.zeros(env.nA)
            new_policy[s][best_action] = 1

        # check if policy has changed
        condition = np.abs(new_policy - policy) == 0
        if np.all(condition):
            # update policy and exit loop
            # don't need to update V
            done = True
            policy = new_policy
            continue
        
        # sanity check: does new policy sum to one?
        normalized = np.sum(policy, axis=1) == 1
        if not np.all(normalized):
            indices = np.argwhere(not normalized)
            raise ValueError('New policy is not normalized for states {}: {}'.format(indices, policy[indices, np.arange(env.nA)]))
        
        # update policy
        policy = new_policy
        
        # update value function
        V = policy_eval_v(policy, env, discount_factor)
    
    ## END MY CODE ##
    
    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    
    ## MY CODE ##
    
    # update until done
    done = False
    
    while not done:
        
        # improve value function by acting greedily w.r.t. to our current guess of Q
        # for this we need to compute the max over actions
        # Q(s,a) = sum_{r,s'} p(s',r|s,a) * (r + \gamma \max_{a'} Q(s',a'))
        new_Q = np.zeros_like(Q)
        
        
        
        for s in range(env.nS):
            
            for a in range(env.nA):                 
                
                for transition in env.P[s][a]:
                    
                    # get transition values
                    prob = transition[0]
                    next_state = transition[1]
                    rew = transition[2]
                    
                    # compute max_{a'} Q(s',a')
                    best_Q = -np.inf
                    for next_a in range(env.nA):
                        if Q[next_state, next_a] > best_Q:
                            best_Q = Q[next_state, next_a]
        
                    # add to new Q-function
                    new_Q[s,a] += prob * (rew + discount_factor * best_Q)
                
            
        # check if stopping criterion is reached, i.e. if max update is smaller than theta
        max_update = np.max(np.abs(Q - new_Q))
        if max_update < theta:
            # we are done after this update
            done = True
            
        # update Q-function
        Q = new_Q
        
        
    
    # compute the optimal policy from Q by computing the argmax over actions
    policy = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)
    policy[np.arange(env.nS), best_actions] = 1
    
    ## END MY CODE ##
    
    return policy, Q
