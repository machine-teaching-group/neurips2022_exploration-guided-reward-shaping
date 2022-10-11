## [NeurIPS 2022] Exploration-Guided Reward Shaping for Reinforcement Learning under Sparse Rewards

### Overview
This folder contains all the code files required for running numerical experiments for tabular settings (Chain and Room environments). The commands below generates data and plots in the folders ```results/runs_chain/```,  ```results/runs_rooom/```, and ```results/plots/```. 


### REINFORCE agent on Chain environment
Run the following command to generate plot for Chain environment without any distractor state

```python teaching_chain_n1_n2.py --agent=reinforce --n2_subgoal=0.0 --n_averaged=10```

Run the following command to generate plot for Chain environment with distractor state

```python teaching_chain_n1_n2.py --agent=reinforce --n2_subgoal=0.01 --n_averaged=10```


### Q-learning agent on Chain environment
Run the following command to generate plot for Chain environment without any distractor state

```python teaching_chain_n1_n2.py --agent=Q_learning --n2_subgoal=0.0 --n_averaged=10```

Run the following command to generate plot for Chain environment with distractor state

```python teaching_chain_n1_n2.py --agent=Q_learning --n2_subgoal=0.01 --n_averaged=10```

### REINFORCE agent on Room environment
Run the following command to generate plot for Room environment without any distractor state

```python teaching_fourroom.py --agent=reinforce --n2_subgoal=0.0 --n_averaged=10```

Run the following command to generate plot for Room environment with distractor state

```python teaching_fourroom.py --agent=reinforce --n2_subgoal=0.01 --n_averaged=10```