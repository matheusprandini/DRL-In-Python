import gym
import numpy as np

def initialize_random_weights():
    return np.random.random(4)*2 - 1

def decide_action(state, weights):
    if state.dot(weights) > 0:
        return 1
    else:
        return 0

def run_one_episode(env, weights):

    total_steps = 0

    observation = env.reset()
    while True:
        total_steps += 1
        #env.render()
        action = decide_action(observation, weights)
        observation, reward, done, _ = env.step(action)
        if done or total_steps == 10000:
            break

    return total_steps

def main():

    env = gym.make('CartPole-v0')

    best_weights = None
    best_weights_steps_mean = None

    for i in range(100):

        print('Starting episode ', i)

        current_weights = initialize_random_weights()
        current_weights_steps = []

        for episode in range(100):

            episode_steps = run_one_episode(env, current_weights)
            current_weights_steps.append(episode_steps)

        current_weights_steps = np.array(current_weights_steps)
        current_weights_steps_mean = np.mean(current_weights_steps)

        print('Episode ', i, ' mean: ', current_weights_steps_mean)

        if (i == 0) or (current_weights_steps_mean > best_weights_steps_mean):
            best_weights = current_weights
            best_weights_steps_mean = current_weights_steps_mean
            print('Updating weights')

    print('Weights', best_weights)
    print('Mean: ', best_weights_steps_mean)

main()
