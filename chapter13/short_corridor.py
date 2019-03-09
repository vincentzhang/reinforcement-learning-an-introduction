#######################################################################
# Copyright (C)                                                       #
# 2018 Sergii Bondariev (sergeybondarev@gmail.com)                    #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def true_value(p):
    """ True value of the first state
    Args:
        p (float): probability of the action 'right'.
    Returns:
        True value of the first state.
        The expression is obtained by manually solving the easy linear system 
        of Bellman equations using known dynamics.
    """
    return (2 * p - 4) / (p * (1 - p))

class ShortCorridor:
    """
    Short corridor environment, see Example 13.1
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0

    def step(self, go_right):
        """
        Args:
            go_right (bool): chosen action
        Returns:
            tuple of (reward, episode terminated?)
        """
        if self.state == 0 or self.state == 2:
            if go_right:
                self.state += 1
            else:
                self.state = max(0, self.state - 1)
        else:
            if go_right:
                self.state -= 1
            else:
                self.state += 1

        if self.state == 3:
            # terminal state
            return -1, True
        else:
            return -1, False

def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)

class ReinforceAgent:
    """
    ReinforceAgent that follows algorithm
    'REINFORCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma, theta=None):
        # set values such that initial conditions correspond to left-epsilon greedy
        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.array([-1.47, 1.47])
        self.alpha = alpha
        self.gamma = gamma
        # first column - left, second - right
        self.x = np.array([[0, 1],
                           [1, 0]])
        self.rewards = []
        self.actions = []

    def get_pi(self):
        h = np.dot(self.theta, self.x)
        pmf = softmax(h)
        # never become deterministic,
        # guarantees episode finish
        imin = np.argmin(pmf)
        epsilon = 0.05

        if pmf[imin] < epsilon:
            pmf[:] = 1 - epsilon
            pmf[imin] = epsilon

        return pmf

    def get_p_right(self):
        return self.get_pi()[1]

    def choose_action(self, reward):
        if reward is not None:
            self.rewards.append(reward)

        go_right = np.random.uniform() <= self.get_p_right()
        self.actions.append(go_right)

        return go_right

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        # a more efficient implementation, adopted in Chapter 5
        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * G[i] * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

class ReinforceAgentOneUpdate(ReinforceAgent):
    def __init__(self, alpha, gamma):
        super(ReinforceAgentOneUpdate, self).__init__(alpha, gamma)

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        # a more efficient implementation, adopted in Chapter 5
        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1
        total_update = 0

        for i in range(len(G)):
            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * G[i] * grad_ln_pi
            total_update += update
            gamma_pow *= self.gamma

        # update only at the end
        self.theta += total_update
        self.rewards = []
        self.actions = []

class ReinforceAgentMultiUpdate(ReinforceAgent):
    def __init__(self, alpha, gamma, num_updates=2):
        super(ReinforceAgentMultiUpdate, self).__init__(alpha, gamma)
        self.num_updates = num_updates

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        # a more efficient implementation, adopted in Chapter 5
        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]


        # update two times
        for k in range(self.num_updates):
            gamma_pow = 1
            for i in range(len(G)):
                j = 1 if self.actions[i] else 0
                pmf = self.get_pi()
                grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
                update = self.alpha * gamma_pow * G[i] * grad_ln_pi

                self.theta += update
                gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        super(ReinforceBaselineAgent, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w
        self.w = 0

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            delta = G[i] - self.w
            self.w += self.alpha_w * gamma_pow * delta

            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * delta * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

class ActorCriticAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        super(ActorCriticAgent, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w
        self.w = 0
        self.gamma_pow = 1

    def choose_action(self, reward):
        if reward is not None:
            # this reward is from the last iteration
            self.rewards.append(reward)
            delta = reward + self.gamma * self.w - self.w
            self.w += self.alpha_w * delta
            j = 1 if self.actions[-1] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * self.gamma_pow * delta * grad_ln_pi
            self.theta += update
            self.gamma_pow *= self.gamma

        go_right = np.random.uniform() <= self.get_p_right()
        self.actions.append(go_right)

        return go_right

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        delta = last_reward - self.w

        self.w += self.alpha_w * delta

        j = 1 if self.actions[-1] else 0
        pmf = self.get_pi()
        grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
        update = self.alpha * self.gamma_pow * delta * grad_ln_pi
        self.theta += update

        self.rewards = []
        self.actions = []
        self.gamma_pow = 1

def trial(num_episodes, agent_generator):
    env = ShortCorridor()
    agent = agent_generator()

    rewards = np.zeros(num_episodes)
    for episode_idx in range(num_episodes):
        rewards_sum = 0
        reward = None
        env.reset()

        while True:
            go_right = agent.choose_action(reward)
            reward, episode_end = env.step(go_right)
            rewards_sum += reward

            if episode_end:
                agent.episode_end(reward)
                break

        rewards[episode_idx] = rewards_sum

    return rewards

def example_13_1():
    epsilon = 0.05
    fig, ax = plt.subplots(1, 1)

    # Plot a graph
    p = np.linspace(0.01, 0.99, 1000)
    y = true_value(p)
    ax.plot(p, y, color='red')

    # Find a maximum point, can also be done analytically by taking a derivative
    imax = np.argmax(y)
    pmax = p[imax]
    ymax = y[imax]
    ax.plot(pmax, ymax, color='green', marker="*", label="optimal point: f({0:.3f}) = {1:.2f}".format(pmax, ymax))

    # Plot points of two epsilon-greedy policies
    ax.plot(epsilon, true_value(epsilon), color='magenta', marker="o", label="epsilon-greedy left")
    ax.plot(1 - epsilon, true_value(1 - epsilon), color='blue', marker="o", label="epsilon-greedy right")

    ax.set_ylabel("Value of the first state")
    ax.set_xlabel("Probability of the action 'right'")
    ax.set_title("Short corridor with switched actions")
    ax.set_ylim(ymin=-105.0, ymax=-10)
    ax.legend()

    plt.savefig('../images/example_13_1.png')
    plt.close()

def figure_13_1():
    num_trials = 100
    num_episodes = 1000
    alphas = [2**-12, 2**-13, 2**-14]
    gamma = 1

    alpha = alphas[0]
    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=gamma)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    plt.plot(np.arange(num_episodes) + 1, rewards.mean(axis=0), color='blue', label=r'$\alpha=2^{-12}$')

    alpha = alphas[1]
    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=gamma)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    plt.plot(np.arange(num_episodes) + 1, rewards.mean(axis=0), color='red', label=r'$\alpha=2^{-13}$')

    alpha = alphas[2]
    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=gamma)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    plt.plot(np.arange(num_episodes) + 1, rewards.mean(axis=0), color='green', label=r'$\alpha=2^{-14}$')

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.savefig('../images/figure_13_1.png')
    plt.close()

# with shaded region
def figure_reinforce():
    num_trials = 100
    num_episodes = 1000
    alphas = [2**-12, 2**-13, 2**-14]
    gamma = 1
    plt.figure()

    alpha = alphas[0]
    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=gamma)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards, color='blue', condition=r'$\alpha=2^{-12}$')

    alpha = alphas[1]
    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=gamma)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards, color='red', condition=r'$\alpha=2^{-13}$')

    alpha = alphas[2]
    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=gamma)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards, color='green', condition=r'$\alpha=2^{-14}$')

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.savefig('../images/figure_13_1.png')
    plt.close()

# with shaded region
def figure_actor_critic():
    num_trials = 10
    num_episodes = 1000
    #alpha = 2**-12
    gamma = 1
    plt.figure()

    alpha_w = 0.1
    alpha_theta = 2**-12
    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ActorCriticAgent(alpha=alpha_theta, gamma=gamma, alpha_w=alpha_w)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards, color='blue', condition=r'$\alpha^{\theta}=2^{-12}, \alpha^{w}=0.1$')

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.savefig('../images/figure_13_actor_critic.png')
    plt.close()

def figure_13_2():
    num_trials = 100
    num_episodes = 1000
    alpha = 2**-13
    alpha_w = 2**-6
    alpha_theta = 2**-9
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceBaselineAgent(alpha=alpha_theta, gamma=gamma, alpha_w=alpha_w)]
    labels = ['Reinforce without baseline',
              'Reinforce with baseline']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_13_2.png')
    plt.close()

def figure_compare():
    num_trials = 100
    num_episodes = 1000
    alpha = 2 ** -13
    alpha_w = 2 ** -6
    alpha_theta = 2 ** -9
    gamma = 1
    agent_generators = [lambda: ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda: ReinforceBaselineAgent(alpha=alpha_theta, gamma=gamma, alpha_w=alpha_w),
                        lambda: ActorCriticAgent(alpha=2**-12, gamma=gamma, alpha_w=0.1)]
    labels = ['Reinforce without baseline',
              'Reinforce with baseline',
              'Actor Critic']

    plt.figure()

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    colors = {labels[0]: 'red', labels[1]: 'green', labels[2]:'blue'}

    for i, label in enumerate(labels):
        sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards[i], color=colors, condition=label)
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_13_compare.png')
    plt.close()

def figure_baseline():
    num_trials = 100
    num_episodes = 1000
    alpha = 2**-13
    alpha_w = 2**-6
    alpha_theta = 2**-9
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceBaselineAgent(alpha=alpha_theta, gamma=gamma, alpha_w=alpha_w)]
    labels = ['Reinforce without baseline',
              'Reinforce with baseline']

    plt.figure()

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    colors = {labels[0]: 'red', labels[1]:'green'}

    for i, label in enumerate(labels):
        sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards[i], color=colors, condition=label)
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_13_2_std.png')
    plt.close()

def figure_one_update_REI():
    num_trials = 100
    num_episodes = 1000
    alpha = 2**-12
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceAgentOneUpdate(alpha=alpha, gamma=gamma)]
    labels = ['Reinforce',
              'Reinforce with one update per episode']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_REI_oneupdate_alpha2-12.png')
    plt.close()

def figure_multi_update_REI():
    num_trials = 100
    num_episodes = 1000
    alpha = 2**-12
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceAgentMultiUpdate(alpha=alpha, gamma=gamma)]
    labels = ['Reinforce',
              'Reinforce with two update per episode']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_REI_twoupdate.png')
    plt.close()

def figure_multi_update_REI_comp():
    num_trials = 100
    num_episodes = 1000
    alpha = 2**-12
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceAgentMultiUpdate(alpha=alpha, gamma=gamma),
                        lambda: ReinforceAgentMultiUpdate(alpha=alpha, gamma=gamma, num_updates=3)]
    labels = ['Reinforce',
              'Reinforce with two updates per episode',
              'Reinforce with three updates per episode']

    #plt.figure()
    #colors = {labels[0]: 'red', labels[1]:'green', labels[2]:'blue', labels[3]:'magenta'}

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
        #sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards[i], color=colors, condition=label)

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_REI_multiupdate.png')
    plt.close()

def figure_three_update_REI():
    num_trials = 100
    num_episodes = 1000
    alpha = 2**-12
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda: ReinforceAgentMultiUpdate(alpha=alpha, gamma=gamma, num_updates=3)]
    labels = ['Reinforce',
              'Reinforce with three updates per episode']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='gray', label='-11.6')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_REI_threeupdates.png')
    plt.close()

def single_exp(theta, fname):
    """ This runs a single experiment for a hyperparam """
    print("Running exp for theta = {}, and save to {}".format(theta, fname))
    num_trials = 100
    num_episodes = 1000
    alpha = 2**-12
    gamma = 1

    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda : ReinforceAgent(alpha=alpha, gamma=gamma, theta=theta)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward

    np.save(fname, rewards)

def run_exp_and_store():
    """ This is the top-level function that runs multiple experiments """
    single_exp(None, 'init_left_eps')
    single_exp(np.array([0.0, 0.0]), 'init_0-0')
    single_exp(np.array([0.0, 1.0]), 'init_0-1')
    single_exp(np.array([1.0, 0.0]), 'init_1-0')

def create_figure():
    labels = ['init with left epsilon (textbook)',
              'init with 0, 0',
              'init with 0, 1',
              'init with 1, 0'
              ]

    experiments = ['init_left_eps', 'init_0-0', 'init_0-1', 'init_1-0']
    temp = np.load('init_left_eps.npy')
    num_trials = temp.shape[0]
    num_episodes = temp.shape[1]

    rewards = np.zeros((len(experiments), num_trials, num_episodes))

    for index, exp_name in enumerate(experiments):
        data = np.load(exp_name+'.npy')
        rewards[index, :, :] = data

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')

    for i, label in enumerate(labels):
        sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards[i], condition=label)
        #plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)

    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.ylim([-95.0, 0])


    plt.savefig('figure_init_comparison.png')

def figure_reinforce_single():
    num_trials = 100
    num_episodes = 100
    alpha = 2**-12
    gamma = 1
    plt.figure()

    rewards = np.zeros((num_trials, num_episodes))
    agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=gamma)

    for i in tqdm(range(num_trials)):
        reward = trial(num_episodes, agent_generator)
        rewards[i, :] = reward
    sns.tsplot(time=np.arange(num_episodes) + 1, data=rewards[:, :num_episodes], color='blue', condition=r'$\alpha=2^{-12}$')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')
    plt.savefig('test_figure.png')
    plt.close()

if __name__ == '__main__':
    #example_13_1()
    #figure_13_1()
    #run_exp_and_store()
    #create_figure()
    #figure_13_2()
    #figure_reinforce()
    #figure_baseline()
    #figure_actor_critic()
    #figure_compare()
    #figure_one_update_REI()
    #figure_multi_update_REI()
    #figure_multi_update_REI_comp()
    figure_three_update_REI()
