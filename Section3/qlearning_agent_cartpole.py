import gym
import numpy as np

env = gym.make('CartPole-v0')
steps = []

class World():

    def __init__(self):
        self.env = gym.make('CartPole-v0').env
        self.create_bins()

    def build_state(self, observation):
        cart_position = observation[0]
        cart_velocity = observation[1]
        pole_angle = observation[2]
        pole_velocity = observation[3]

        cart_position_bin = self.find_bin(cart_position, self.cart_position_bins)
        cart_velocity_bin = self.find_bin(cart_velocity, self.cart_velocity_bins)
        pole_angle_bin = self.find_bin(pole_angle, self.pole_angle_bins)
        pole_velocity_bin = self.find_bin(pole_velocity, self.pole_velocity_bins)

        features = [cart_position_bin, cart_velocity_bin, pole_angle_bin, pole_velocity_bin]

        return int("".join(map(lambda feature: str(int(feature)), features)))

    def create_bins(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-1, 1, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def find_bin(self, value, feature_bin):
        return np.digitize(x=[value], bins=feature_bin)[0]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class Agent():

    def __init__(self):
        self.q_table = self.initialize_q_table()
        self.number_episodes = 10000
        self.epsilon = 1.
        self.epsilon_decay = .001
        self.gamma = .9
        self.learning_rate = .1

    def initialize_q_table(self):
	## 10000 states and 2 possible actions
        return np.zeros((10000, 2))

    def execute_episode_with_epsilon_greedy(self, env):

        # Reseting the initial state
        observation = env.reset()
        done = False
        num_steps = 0

	# Buildind current state
        current_state = env.build_state(observation)

	# Until the end of the episode
        while True:

	    # Generate a random float number between 0 and 1
            random_number = np.random.uniform(0,1)
			
	    # Exploration -> random choice (action)
            if random_number <= self.epsilon:
                action = np.random.choice([0,1])
            # Exploitation -> best action based on q_value
            else:
                action = self.q_table[current_state].argmax(axis=0)

            # Execute the action and reach the new_state
            observation, immediate_reward, done, _ = env.step(action)	
            new_state = env.build_state(observation)
            
	    # Verify the best action on the new_state (higher q-value)
            best_q_value = np.amax(self.q_table[new_state])

            # Incrementing num_steps
            num_steps += 1

            # Checking terminal states
            if done:
                immediate_reward = -300.0

            if num_steps >= 10000:
                immediate_reward = 300.0
			
            # Update Q-Table
            self.q_table[current_state][action] = ((1 - self.learning_rate) * self.q_table[current_state][action]) + (self.learning_rate * (immediate_reward + (self.gamma * best_q_value)))

            # Update the current state
            current_state = new_state

	    # Checking loop condition
            if done or num_steps >= 10000:
                break

        return num_steps

    def training_q_learning(self, env):

        all_episodes_steps = []

        for i in range(self.number_episodes):
            steps = self.execute_episode_with_epsilon_greedy(env)
            all_episodes_steps.append(steps)

            print("Episode ", i, ': ', steps, ' steps')

            if self.epsilon > 0.1:
                self.epsilon -= self.epsilon_decay

        print('Average steps for the last 100 episodes: ', np.array([all_episodes_steps[-100:]]).mean())


cartpole = World()
agent = Agent()

agent.training_q_learning(cartpole)


