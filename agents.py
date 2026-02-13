import numpy as np

class BanditAgent:
    """
    Parent class for an agent that learns to select actions in a k-armed bandit problem.
    """
    
    def __init__(self, k: int, step_size: float | None = None, initial_value: float = 0.0):
        """
        Initialize the bandit agent.
        
        Parameters:
        -----------
        k : int
            Number of arms
        initial_value : float
            Initial estimate Q_1(a) for all actions
        """
        self.k = k
        self.step_size = step_size

        self.q_vals = np.full(self.k, initial_value)
        self.action_counts = np.zeros(shape=self.q_vals.shape)
        
    
    def select_action(self) -> int:
        """
        Virtual function to Select an action using any algorithm
        
        Returns:
        --------
        action : int
            Selected action (0 to k-1)
        """
        raise NotImplementedError()
    
    def update(self, action: int, reward: float):
        """
        Update action-value estimates after receiving a reward.
        
        Parameters:
        -----------
        action : int
            The action that was taken
        reward : float
            The reward that was received
        """
        self.action_counts[action] += 1
        
        # Q_n+1 = Q_n + α[R_n - Q_n]
        # If step_size is None, use α = 1/N(action) (sample-average)
        # If step_size is set, use that constant value
        if self.step_size:
            alpha = self.step_size
        else: # step_size = None
            alpha = 1/self.action_counts[action]

        self.q_vals[action] = self.q_vals[action] + (alpha * (reward - self.q_vals[action]))
        
        
    def reset(self, initial_value: float = 0.0):
        """Reset the agent for a new run."""
        # TODO: Re-initialize action-value estimates and counts
        self.q_vals = np.full(self.k, initial_value)
        self.action_counts = np.zeros(self.k)
    
    def debug(self):
        print("q_vals:", self.q_vals)
        print("action_counts:", self.action_counts)
        print("step_size:", self.step_size)
    
    def algorithm_name(self):
        return NotImplementedError()
    
    def algorithm_metric(self):
        return NotImplementedError()


class GreedyBanditAgent(BanditAgent):
    def __init__(self, k, step_size = None, initial_value = 0, epsilon: float = 0.1):
        super().__init__(k, step_size, initial_value)
        self.epsilon = epsilon

    def select_action(self):
        """
        Select an action with episilon greedy policy

        Returns:
        --------
        action : int
            Selected action (0 to k-1)
        """

        explore = True if np.random.uniform(low=0, high=1, size=1) < self.epsilon else False

        if explore:
            # Exploration: random action
            action = np.random.randint(0, self.k)
        else:
            # Exploitation: greedy action
            best_action = int(np.argmax(self.q_vals))
            action = best_action
        
        return action

    def algorithm_name(self):
        return "Epsilon-Greedy"
    
    def algorithm_metric(self):
        return "epsilon", self.epsilon
    

class SoftmaxBanditAgent(BanditAgent):
    def __init__(self, k, step_size: float | None = None, initial_value: float = 0, temp: float = 0.1):
        super().__init__(k, step_size, initial_value)
        self.temp = temp

    def select_action(self):
        """
        Select an action using softmax policy.

        Returns:
        --------
        action : int
            Selected action (0 to k-1)
        """

        Q = self.q_vals
        Q_max = max(self.q_vals)
        temp = self.temp

        P_arm_vals = np.zeros_like(Q)

        # Calculate softmax probabilities using numerically stable method
        exp_vals = np.exp((Q - Q_max) / temp)
        P_arm_vals = exp_vals / np.sum(exp_vals)
        self.P_arm_vals = P_arm_vals
        action = np.argmax(P_arm_vals)

        return action

    def algorithm_name(self):
        return "Softmax"
    
    def algorithm_metric(self):
        return "temp", self.temp
    

class UCBBanditAgent(BanditAgent):
    def __init__(self, k, step_size: float | None = None, initial_value: float = 0, c: float = 1):
        super().__init__(k, step_size, initial_value)
        self.c = c

    def select_action(self):
        """
        Select an action using softmax policy.

        Returns:
        --------
        action : int
            Selected action (0 to k-1)
        """

        Q = np.array(self.q_vals) # 1d np array
        N = np.array(self.action_counts) # 1d np array
        t = np.sum(N)
        c = self.c 

        variance = np.zeros_like(N)
        with np.errstate(divide='ignore', invalid='ignore'): # Handle N_t(a) = 0
            # variance = c * np.sqrt(np.log(t) / N_t(a))
            variance = c * np.sqrt(np.log(t) / N)
            variance[N == 0] = np.inf

    
        A_t = np.argmax(Q + variance)
        return A_t
    
    def algorithm_name(self):
        return "UCB"
    
    def algorithm_metric(self):
        return "c", self.c
  
    
if __name__ == "__main__":
    # Test Agents
    ubc_agent = UCBBanditAgent(4, initial_value=5, c = 0.1)
    ubc_agent.update(0, 10)
    ubc_agent.update(1, 1) 
    ubc_agent.update(0, 5)
    ubc_agent.select_action()