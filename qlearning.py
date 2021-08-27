import math

import gym
import matplotlib.pyplot as plt
import numpy as np


class QLearningSolver():
    def __init__(self, buckets=(1, 1, 1, 1), n_iterations=1000, max_episodes=200, decay_rate=0.01):
        self.buckets = buckets
        self.decay_rate = decay_rate
        self.gamma = 1.0
        self.min_alpha = 0.1
        self.min_epsilon = 0.1
        self.n_iterations = n_iterations
        self.max_episodes = max_episodes

        self.env = gym.make('CartPole-v0')

        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    # generate a random number, if this is less than epsilon (exploration policy),
    # then return a random action (1, 0). If greather than epsilon, return value
    # from Q Table
    def choose_action(self, state, epsilon):
        n_random = np.random.random()

        if n_random <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    # lower and upper bounds are set by openai gym
    # calculate no of intervals by taking delta bound divided by no of buckets
    # check each variable is not less than or greater than bounds, if true, set to bound
    # calculate each bucket, if value if within bucket, return that bucket
    def discretise(self, obs):
        lower_bounds = [-4.8, -5.0, -0.418, -5.0]
        upper_bounds = [4.8, 5.0, 0.418, 5.0]

        intervals = [(upper_bounds[k] - lower_bounds[k]) / max(v - 1, 1) for k, v in enumerate(self.buckets)]

        new_obs = [0, 0, 0, 0]

        for k, v in enumerate(obs):
            if v < lower_bounds[k]:
                v = lower_bounds[k]

            if v > upper_bounds[k]:
                v = upper_bounds[k]

            for i in range(self.buckets[k]):
                low = lower_bounds[k] + (i * intervals[k])
                high = low + intervals[k]

                if v >= low and v <= high:
                    new_obs[k] = i

                    break

        return tuple(new_obs)

    # exponential decay algorithm
    # t - current iteration
    def exp_decay(self, t):
        # using max() sets a lower limit of 0.1
        return max(0.1, 1.0 * math.pow(1 - self.decay_rate, t))

    def run(self):
        # arrays to track results through every iteration
        all_pole_angles = []
        all_scores = []

        for iteration in range(self.n_iterations):
            current_state = self.env.reset()
            current_state = self.discretise(current_state)
            done = False

            # calculate new values for exploration policy and learning rate
            alpha = self.exp_decay(iteration)
            epsilon = self.exp_decay(iteration)

            # keep track of current iteration performance
            pole_angles = []
            score = 0

            while not done:
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _  = self.env.step(action)

                new_state = self.discretise(obs)
                self.update_q(current_state, new_state, action, alpha, reward)
                current_state = new_state

                pole_angles.append(obs[2])
                score += reward

            all_pole_angles.append(pole_angles)
            all_scores.append(score)

            max_score = np.max(all_scores)

            # if max score is greater than or equal to win condition, finish
            if max_score >= self.max_episodes:
                print(f'Solved after {iteration} iterations.')

                return all_pole_angles, all_scores

            # print to console current status on every 50 iterations
            if iteration % 50 == 0:
                print(f'[Iteration {iteration}] - Max survival time is {max_score}.')

        # never completed
        print(f'Did not solve after {iteration} iterations.')

        return all_pole_angles, all_scores

    # bellman's equation for q-learning
    def update_q(self, current_state, new_state, action, alpha, reward):
        self.Q[current_state][action] += alpha * (reward + self.gamma * np.max(self.Q[new_state]) - self.Q[current_state][action])


if __name__ == '__main__':
    solver = QLearningSolver(buckets=(1, 1, 12, 6))
    all_pole_angles, all_scores = solver.run()

    plt.figure(1)
    plt.grid()
    plt.plot(all_scores)
    plt.title('Max. Episode per Iteration')
    plt.xlabel('Iteration')
    plt.xlim(left=0)
    plt.ylabel('Episode')
    plt.ylim(bottom=0, top=200)

    plt.figure(2)
    plt.grid()
    for _, v in enumerate(all_pole_angles):
        plt.plot(v)
    plt.title('Pole Angles over Timestep')
    plt.xlabel('Timestep')
    plt.xlim(left=0, right=200)
    plt.ylabel('Pole Angle [Radians]')
    plt.ylim(bottom=-0.4, top=0.4)

    plt.show()
