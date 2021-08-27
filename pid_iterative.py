import gym
import matplotlib.pyplot as plt
import numpy as np


class PIDIterativeSolver():
    def __init__(self, pid_min=0, pid_max=0, pid_delta=0, max_episodes=200):
        self.max_episodes = max_episodes
        self.pid_opts = []
        for p in range(pid_min, pid_max, pid_delta):
            for i in range(pid_min, pid_max, pid_delta):
                for d in range(pid_min, pid_max, pid_delta):
                    self.pid_opts.append({'p': p, 'i': i, 'd': d})

        self.env = gym.make('CartPole-v0')

    # sigmoid function
    def choose_action(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def run(self):
        # arrays to track results through every iteration
        all_pid_values = [[], [], []]
        all_pole_angles = []
        all_scores = []

        for k, v in enumerate(self.pid_opts):
            current_state = self.env.reset()
            done = False

            # get pid values
            p = v['p']
            i = v['i']
            d = v['d']

            # pid calculation variables
            integral = 0
            derivative = 0
            prev_error = 0

            # add current p, i, d to tracking array
            all_pid_values[0].append(p)
            all_pid_values[1].append(i)
            all_pid_values[2].append(d)

            # keep track of current iteration performance
            pole_angles = []
            score = 0

            while not done:
                pole_angle = current_state[2]

                integral += pole_angle
                derivative = pole_angle - prev_error
                prev_error = pole_angle

                pid = p * pole_angle + i * integral + d * derivative

                # get action from sigmoid function
                # parse to int (0 or 1)
                action = self.choose_action(pid)
                action = np.round(action).astype(np.int32)
                obs, reward, done, _  = self.env.step(action)

                current_state = obs

                pole_angles.append(obs[2])
                score += reward

            all_pole_angles.append(pole_angles)
            all_scores.append(score)

            max_score = np.max(all_scores)

            # print to console current status on every 50 iterations
            if k % 50 == 0:
                print(f'[Iteration {k} of {len(self.pid_opts)}] - Max survival time is {max_score}.')

        return all_pid_values, all_pole_angles, all_scores


if __name__ == '__main__':
    solver = PIDIterativeSolver(pid_min=1, pid_max=20, pid_delta=1)
    all_pid_values, all_pole_angles, all_scores = solver.run()

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

    plt.figure(3)
    plt.grid()
    plt.plot(all_pid_values[0], label='P')
    plt.plot(all_pid_values[1], label='I')
    plt.plot(all_pid_values[2], label='D')
    plt.title('PID Values over Iteration')
    plt.xlabel('Iteration')
    plt.xlim(left=0)
    plt.legend()

    plt.show()
