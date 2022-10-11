## [NeurIPS 2022] Exploration-Guided Reward Shaping for Reinforcement Learning under Sparse Rewards

### Overview
This folder contains all the code files required for running numerical experiments for neural settings (LineK environment). The commands below generates data and plots in the folders ```results/runs_lineKey/``` and ```results/plots/```. 

### REINFORCE agent on LineK environment
Run the following command to generate plot for LineK environment without any distractor state

```python teaching_linekeymulti.py --n_picks=10 --epsilon_reinforce=0.05 --small_reward_for_goal_without_key=0.0 --n_averaged=10 ```

Run the following command to generate plot for LineK environment with distractor states

```python teaching_linekeymulti.py --n_picks=10 --epsilon_reinforce=0.05 --small_reward_for_goal_without_key=0.01 --n_averaged=10 ```
