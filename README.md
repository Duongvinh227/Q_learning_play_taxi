# Q_learning_play_taxi
The Taxi Problem from “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition” by Tom Dietterich

![image](https://github.com/Duongvinh227/Q_learning_play_taxi/assets/96807833/cd3ab051-a6fd-4ee8-8948-53a6438251e2)


The general formula of the provided code snippet is:
Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', :)) - Q(s, a))

Where:

- Q(s, a) is the value of the Q-function for state s and action a.
- α is the learning rate for updating the Q-values.
- r is the reward received after taking action a in state s.
- γ is the discount factor, ranging from 0 to 1, which determines the importance of future rewards compared to the immediate reward.
- max(Q(s', :)) is the maximum value among all possible actions that can be taken from the next state s'.
- Q(s', :) is the set of Q-values for all possible actions that can be taken from state s'.
- 
This is the Q-value update formula in the Q-learning algorithm, where the goal is to gradually improve the Q-values to achieve optimal actions for each state.


