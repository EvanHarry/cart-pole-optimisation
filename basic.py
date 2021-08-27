import gym
import matplotlib.pyplot as plt
import numpy as np


class BasicSolver():
    def __init__(self, n_iterations=1000, max_episodes=200):
        self.n_iterations = n_iterations
        self.max_episodes = max_episodes

        self.env = gym.make('CartPole-v0')

    # move cart left (0) if pole angle is less than 0
    # move cart right (1) if pole angle is greater than 1
    def choose_action(self, state):
        pole_angle = state[2]

        return 0 if pole_angle < 0 else 1

    def run(self):
        # arrays to track results through every iteration
        all_pole_angles = []
        all_scores = []

        for iteration in range(self.n_iterations):
            current_state = self.env.reset()
            done = False

            # keep track of current iteration performance
            pole_angles = []
            score = 0

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _  = self.env.step(action)

                current_state = obs

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


if __name__ == '__main__':
    solver = BasicSolver()
    all_pole_angles, all_scores = solver.run()

    plt.figure(1)
    plt.grid()
    plt.plot(all_scores)
    plt.title('Max. Episode per Iteration')
    plt.xlabel('Iteration')
    plt.xlim(left=0, right=1000)
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
