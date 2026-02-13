import numpy as np
import matplotlib.pyplot as plt
from rlcmab_sampler import sampler

class MultiArmedBandit:
    """
    A k-armed bandit environment for reinforcement learning experiments.
    
    This class represents a bandit problem where each arm has a true value q*(a)
    and rewards are sampled from a normal distribution around that value.
    """
    
    def __init__(self, k: int = 12):
        self.k = k
        self.reward_sampler = sampler(102)
        
    
    def pull(self, action: int) -> float:
        reward = self.reward_sampler.sample(action)
        return reward
    


def plot_arm_reward_distributions(bandit: MultiArmedBandit, 
                                  samples_per_arm: int = 200) -> None:

    # TODO 1: Create a (samples_per_arm x k) matrix to store rewards
    results = np.zeros(shape=(samples_per_arm, bandit.k))

    # TODO 2: Collect samples_per_arm rewards from each arm using bandit.pull(a)
    for sample_index in range(samples_per_arm):
        for arm_index in range(bandit.k):
            results[sample_index, arm_index] = bandit.pull(arm_index)

    # TODO 3: Compute empirical mean reward for each arm (μ̂)
    empirical_means = results.mean(axis=0) # axis = 0 => down each col

    # TODO 5: Plot violin plots (one violin per arm)
    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(results, showmeans=False, showmedians=True, showextrema=False)

    # Overlay empirical means (μ̂) and true means (q*)
    ax.scatter(np.arange(1, bandit.k + 1), empirical_means, color='blue', label='Empirical Mean (μ̂)', zorder=3)

    ax.set_title("Reward Distributions for Each Arm")
    ax.set_xlabel("Arm")
    ax.set_ylabel("Reward")
    ax.grid("on")
    ax.set_xticks(np.arange(1, bandit.k + 1))
    ax.legend()
    plt.show()


if __name__ == "__main__":
    bandit = MultiArmedBandit(k=12)
    plot_arm_reward_distributions(bandit, samples_per_arm=200)

