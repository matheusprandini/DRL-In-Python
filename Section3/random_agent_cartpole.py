import gym
import numpy as np

env = gym.make('CartPole-v0')
steps = []

for episode in range(100):
    observation = env.reset()
    num_steps = 0
    while True:
        num_steps += 1
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('Number of steps: ', num_steps)
            steps.append(num_steps)
            break

steps = np.array(steps)
print('Steps : ', steps)
print('Median: ', np.mean(steps))
