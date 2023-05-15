import time
import gym
import numpy as np

env = gym.make('Taxi-v3' , render_mode="human")

alpha = 0.5  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.01  # exploration rate
max_epsilon = 1.0  # maximum exploration rate
min_epsilon = 0.01  # minimum exploration rate
decay_rate = 0.9  # exploration decay rate

counter = 0

num_states = env.observation_space.n
num_actions = env.action_space.n
# [500 , 6]
Q = np.zeros((num_states, num_actions))

num_episodes = 10000000
max_steps_per_episode = 30

for episode in range(num_episodes):
    counter = counter + 1

    state = env.reset()[0]

    for step in range(max_steps_per_episode):

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, _, info = env.step(action)

        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        state = new_state
        # env.render()
        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    print("Epsilon:", epsilon, "Episode :", episode, "Reward :", reward)

np.save(file="Q_table.npy", arr=Q)
print()
