# Lab 3: Contextual Bandit-Based News Article Recommendation

## Overview
This project is part of the Reinforcement Learning Fundamentals course. It focuses on implementing a contextual bandit-based recommendation system for news articles. The system uses user and article data to recommend relevant articles to users.

## Project Structure
- **`master.ipynb`**: The main notebook for the project, including data loading, preprocessing, and contextual bandit implementation.
- **`classifiers.py`**: Includes helper functions and classes for classification tasks.
- **`bandit.py`**: Contains bandit base class called `MultiArmedBandit` which calls `rlcmab_sampler` to sample rewards.
- **`agents.py`**: Contans all the agent base classes implements $\epsilon$-Greedy, Softmax, and UCB algorithms.
- **`experiments.py`**: Functions to run multiple experiments and plot results.