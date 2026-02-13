import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agents import *
from bandit import * 
from typing import Tuple, Type, List
from tqdm import tqdm

def run_experiment(bandit: MultiArmedBandit, 
                   agent: BanditAgent, 
                   steps: int,
                   action_offset: int) -> np.ndarray:
    """
    Run a single experiment: agent interacting with bandit.
    
    action_offset defines how much is the offset to match the arm in the actual bandit

    Returns:
    --------
    rewards : np.ndarray
        Reward at each time step
    """
    rewards = np.zeros(steps)
    
    for t in range(steps):
        # Select an action
        action_t = agent.select_action()
        # Get reward
        reward_t = bandit.pull(action_t + action_offset)
        # Update Q function using the reward
        agent.update(action_t, reward_t)
        
        # Record results
        rewards[t] = reward_t

    return rewards


def run_multiple_experiments(k: int = 12, 
                             agent_type: Type[BanditAgent] = GreedyBanditAgent,
                             metric: float = 0.1,
                             step_size: float | None = None,
                             initial_value: float = 0,
                             steps: int = 10000,
                             runs: int = 2000) -> np.ndarray:
    """
    Run multiple MAB runs and return average rewards
    """

    # k = 12 and there are 3 users 
    # Each agent only looks at a 4 arms each
    n_users = 3
    n_arms = int(k / n_users)

    # Each run will have n_users number of agents.
    runs = int(runs / n_users)

    all_rewards = np.zeros((runs, steps))
    
    for run in tqdm(range(runs), "Run"):
        # Create new bandit and agent for each run
        bandit = MultiArmedBandit(k)

        for i in range(n_users):
            action_offset = i * n_arms
            agent = agent_type(n_arms, step_size, initial_value, metric)

            # Run experiment
            rewards = run_experiment(bandit, agent, steps, action_offset)
            all_rewards[run] = rewards
            
    # Average across all runs
    avg_rewards = all_rewards.mean(axis=0)
    return avg_rewards


def run_agent_comparison(agent_type: Type[BanditAgent],
                         metrics: List[float],
                         k: int = 12,
                         step_size: float | None = None,
                         initial_value: float = 0,
                         steps: int = 1000, 
                         runs: int = 2000) -> None:
    """
    Run experiments comparing different hyperparameter values for a given agent type.
    Plots average reward vs steps for each metric value.

    Parameters:
    -----------
    agent_type : Type[BanditAgent]
        The agent class to test (e.g., GreedyBanditAgent, SoftmaxBanditAgent, UCBBanditAgent)
    metrics : List[float]
        List of hyperparameter values to compare (e.g., epsilon values, temperature values, or c values)
    k : int
        Number of arms in the bandit
    step_size : float | None
        Step size for Q-value updates (None for sample average)
    initial_value : float
        Initial Q-value estimates
    steps : int
        Number of time steps per run 
    runs : int
        Number of independent bandit runs
    """

    avg_rewards_dict = {}

    # Run experiments for each metric value
    for metric_val in metrics:
        avg_reward = run_multiple_experiments(
            k=k, 
            agent_type=agent_type,
            metric=metric_val, 
            step_size=step_size,
            initial_value=initial_value,
            steps=steps, 
            runs=runs
        )
        avg_rewards_dict[metric_val] = avg_reward

    # Get agent name and metric name for titles/labels
    # Create a temporary agent instance to get the algorithm name and metric name
    temp_agent = agent_type(k, step_size, initial_value, metrics[0])
    algo_name = temp_agent.algorithm_name()
    metric_name, _ = temp_agent.algorithm_metric()

    # Plot average reward vs steps
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric_val in metrics:
        ax.plot(avg_rewards_dict[metric_val], label=f"{metric_name} = {metric_val}")
    
    ax.set_title(f"{algo_name} Agent: Average Reward vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":    
    # Example: Compare softmax with different temperature values
    run_agent_comparison(
        agent_type=SoftmaxBanditAgent,
        metrics=[0.1, 0.5, 1.0],
        initial_value=3,
        steps=400,
        runs=500
    )
    
    # Example: Compare UCB with different c values
    # run_agent_comparison(
    #     agent_type=UCBBanditAgent,
    #     metrics=[0.5, 1.0, 2.0],
    #     steps=1_000,
    #     runs=500
    # )