#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 12

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# gamma for Q-Learning and Expected Sarsa
GAMMA = 1

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
START = [3, 0]
GOAL = [3, 11]


# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)

class CliffWalk(object):
    """
    CliffWalk environment, see Example 6.6
    """
    def __init__(self):
        self.reset()
        self.actions = ACTIONS

    def reset(self):
        self.state = START
        return self.state

    def num_states(self):
        return WORLD_HEIGHT*WORLD_WIDTH

    def num_actions(self):
        return len(self.actions)

    def step(self, action):
        is_terminal = False
        i, j = self.state
        if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
                action == ACTION_RIGHT and self.state == START):
            reward = -100
            self.state = START
            return self.state, reward, is_terminal

        if action == ACTION_UP:
            self.state = [max(i - 1, 0), j]
        elif action == ACTION_LEFT:
            self.state = [i, max(j - 1, 0)]
        elif action == ACTION_RIGHT:
            self.state = [i, min(j + 1, WORLD_WIDTH - 1)]
        elif action == ACTION_DOWN:
            self.state = [min(i + 1, WORLD_HEIGHT - 1), j]
        else:
            assert False

        reward = -1
        if self.state == GOAL:
            is_terminal = True
        return self.state, reward, is_terminal

def trial(num_episodes, agent_generator):
    env = CliffWalk()
    agent = agent_generator()
    agent.init_param(env.num_states(), env.num_actions())

    rewards = np.zeros(num_episodes)
    steps = np.zeros(num_episodes)

    for episode_idx in range(num_episodes):
        rewards_sum = 0
        num_steps = 0
        reward = None
        state = env.reset()

        while True:
            action = agent.choose_action(state, reward)
            next_state, reward, episode_end = env.step(action)
            rewards_sum += reward
            num_steps += 1

            if episode_end:
                agent.episode_end(reward)
                break
            else:
                agent.update(next_state, reward)
            state = next_state # update the current state for next step

        rewards[episode_idx] = rewards_sum
        steps[episode_idx] = num_steps

        print("Reward/Step at Episode {} is {}/{}".format(episode_idx, rewards_sum, num_steps))

    return rewards, steps

class ReinforceAgent(object):
    """
    ReinforceAgent that follows algorithm
    'REINFORCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma, theta=None, num_states=None, num_actions=None):
        # set values such that initial conditions correspond to left-epsilon greedy
        if theta is not None:
            self.theta = theta
            # theta is [1x(SxA)] vector
        else:
            if (num_states is not None) and (num_actions is not None):
                self.theta = np.zeros((1, num_states * num_actions))
                self.num_actions = num_actions
                self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        # first column - action 0, second - action 1, ...
        # make it more general, it needs an input of the number of policy parameters
        if num_actions is not None:
            self.x = np.zeros((self.theta.shape[1], num_actions))
        self.rewards = []
        self.actions = []
        self.states = []

    def init_param(self, num_states, num_actions):
        self.theta = np.zeros((1, num_states * num_actions))
        self.x = np.zeros((self.theta.shape[1], num_actions))
        self.num_actions = num_actions
        self.num_states = num_states

    def update(self, next_state, reward):
        # callback for performing update based on the next state,reward pair
        pass

    def state_to_idx(self, state):
        i, j = state
        state_idx = i * WORLD_WIDTH + j
        return state_idx

    def update_x(self, state):
        state_idx = self.state_to_idx(state)
        self.x[:] = 0.0
        for action_idx in range(self.num_actions):
            self.x[state_idx + self.num_states * action_idx, action_idx] = 1.0

    def get_pi(self):
        h = np.dot(self.theta, self.x) # theta: 1xSA, self.x: SAxA
        pmf = softmax(h)
        # never become deterministic,
        # guarantees episode finish
        imin = np.argmin(pmf)
        epsilon = EPSILON
        #
        # print("imin is: ", imin)
        # print("pmf[imin] is ", pmf[imin])
        # print("h is", h)
        # print("pmf[:] is ", pmf[:])
        imax = np.argmax(pmf)

        if pmf[0, imin] < epsilon:
            pmf[:] = (1 - epsilon)/(self.num_actions-1)
            # print("epsilone is ", epsilon)
            # print("pmf[:] is ", pmf[:])
            pmf[0, imin] = epsilon

            # Another approach, set all probs below epsilon as epsilon,
            # and set all the others according to their original proportion
            # if pmf[0, imax] > 1-epsilon:
            #     pmf[:] = epsilon/(self.num_actions-1)
            #     pmf[0, imax] = 1-epsilon
            # else:
            #     num_actions_smaller_prob = (pmf <= epsilon).sum()
            #     pmf[pmf <= epsilon] = epsilon/num_actions_smaller_prob
            #     sum_actions_larger_prob = pmf[pmf > epsilon].sum()
            #     pmf[pmf > epsilon] = pmf[pmf > epsilon] / sum_actions_larger_prob * (1-epsilon)

        return pmf

    def choose_action(self, state, reward):
        if reward is not None:
            self.rewards.append(reward)

        # not picking action from Q-values anymore. pick from policy parameters
        self.states.append(state)
        # first update x based on the state
        self.update_x(state)
        action = np.random.choice(ACTIONS, p=self.get_pi()[0,:])
        self.actions.append(action) # need to memorize all the actions in this episode
        return action

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
            j = self.actions[i]
            state = self.states[i]
            self.update_x(state)
            pmf = self.get_pi() # 1xA
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf[0,:])
            update = self.alpha * gamma_pow * G[i] * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []
        self.states = []

class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        super(ReinforceBaselineAgent, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w

    def init_param(self, num_states, num_actions):
        self.theta = np.zeros((1, num_states * num_actions))
        self.x = np.zeros((self.theta.shape[1], num_actions))
        self.num_actions = num_actions
        self.num_states = num_states
        # w should be state-dependent, 48 states
        self.w = np.zeros((1, self.num_states)) # v(St, w)

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            state = self.states[i]
            state_idx = self.state_to_idx(state)

            delta = G[i] - self.w[0, state_idx]

            self.w[0, state_idx] += self.alpha_w * delta

            j = self.actions[i]

            self.update_x(state)
            pmf = self.get_pi() # 1xA
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf[0,:])
            update = self.alpha * gamma_pow * delta * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []
        self.states = []

class RandomAgent(object):
    '''
        Random Agent that performs equiprobable
    '''

    def __init__(self, alpha, gamma, num_states=None, num_actions=None):
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        self.actions = []
        self.state = None

    def init_param(self, num_states, num_actions):
        pass

    def update(self, next_state, reward):
        # callback for performing update based on the next state,reward pair
        pass

    def choose_action(self, state, reward):
        self.state = state
        action = np.random.choice(ACTIONS)
        self.actions.append(action) # need to memorize all the actions in this episode
        return action

    def episode_end(self, last_reward):
        self.actions = []
        self.state = None

class QLearningAgent(object):
    """
    QLearning Agent that follows Q-learning algorithm
    """
    def __init__(self, alpha, gamma, num_states=None, num_actions=None):

        self.alpha = alpha
        self.gamma = gamma
        self.q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        self.num_actions = num_actions
        self.num_states = num_states
        self.actions = []
        self.state = None

    def init_param(self, num_states, num_actions):
        pass

    def update(self, next_state, reward):
        # callback for performing update based on the next state,reward pair
        self.q_values[self.state[0], self.state[1], self.actions[-1]] += self.alpha * (
                reward + self.gamma * np.max(self.q_values[next_state[0], next_state[1], :]) -
                self.q_values[self.state[0], self.state[1], self.actions[-1]])

    def choose_action(self, state, reward):
        #if reward is not None:
        #    self.rewards.append(reward)

        self.state = state

        action = choose_action(state, self.q_values)
        self.actions.append(action) # need to memorize all the actions in this episode
        return action

    def episode_end(self, last_reward):
        self.actions = []
        self.state = None

def cliffwalk_pg():
    ''' This function runs the REINFORCE algorithm on cliffwalk '''
    # episodes of each run
    episodes = 100

    # perform 50 independent runs
    runs = 50

    # settings of the REINFORCE agent
    #alphas = [2**-8, 2**-10, 2**-12, 2**-14, 2**-16, 2**-18, 2**-20, 2**-22]
    alphas = [2**-20]
    hyperparamsearch = False

    for alpha in alphas:
        rewards_rei = np.zeros((runs, episodes))
        steps_rei = np.zeros((runs, episodes))
        agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=GAMMA)

        #rewards_sarsa = np.zeros((runs, episodes))
        #rewards_q_learning = np.zeros((runs, episodes))
        for r in tqdm(range(runs)):
            #q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
            #q_q_learning = np.copy(q_sarsa)
            #for i in range(0, episodes):
                #rewards_sarsa[r, i] = sarsa(q_sarsa)
                #rewards_q_learning[r, i] = q_learning(q_q_learning)
                rewards_rei[r, :], steps_rei[r, :] = trial(episodes, agent_generator)

        # stats of rewards, write to a txt file
        sum_rewards = rewards_rei.sum(axis=1).mean()  # this is the sum over episodes, averaged of the runs
        print("alpha: {}".format(alpha))
        print("sum of rewards: {}".format(sum_rewards))
        if hyperparamsearch:
            file = open("log/sum_rewards_alpha_{}_rewards_{}.txt".format(alpha, sum_rewards), "w")
            file.write("sum of rewards: {} \n".format(sum_rewards))
            file.write(
                'The Min/Max Reward in one episode is {}/{}\n'.format(rewards_rei.min(), rewards_rei.max()))
            file.write('The Max/Min Steps is {}/{}'.format(steps_rei.max(), steps_rei.min()))
            file.close()

    # draw reward curves
    if not hyperparamsearch:
        plt.figure()
        sns.tsplot(data=rewards_rei, color='blue', condition='REINFORCE')
        #plt.plot(rewards_rei.mean(axis=0), label='REINFORCE')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        #plt.ylim([-100, 0])
        plt.legend()
        plt.savefig('../images/figure_6_4_pg.png')
        plt.close()

def cliffwalk_pg_baseline():
    ''' This function runs the REINFORCE-baseline algorithm on cliffwalk '''
    # episodes of each run
    episodes = 5

    # perform 50 independent runs
    runs = 5

    # settings of the REINFORCE agent
    alphas = [2**-10, 2**-12, 2**-14, 2**-16, 2**-18, 2**-20, 2**-22]
    #alpha = alphas[0]
    alpha_ws = [2**-4, 2**-5, 2**-6]
    #alpha_w = 2**-5

    for alpha in alphas:
        for alpha_w in alpha_ws:
            rewards_baseline = np.zeros((runs, episodes))
            steps_baseline = np.zeros((runs, episodes))

            agent_generator = lambda: ReinforceBaselineAgent(alpha=alpha, gamma=GAMMA, alpha_w=alpha_w)

            for r in tqdm(range(runs)):
                rewards_baseline[r, :], steps_baseline[r, :] = trial(episodes, agent_generator)

            # stats of rewards, write to a txt file
            sum_rewards = rewards_baseline.sum(axis=1).mean() # this is the sum over 200 episodes, averaged of 50 runs
            file = open("log/sum_rewards_alpha_{}_alphaw_{}.txt".format(alpha, alpha_w), "w")
            file.write("sum of rewards: {} \n".format(sum_rewards))
            file.write('The Min/Max Reward in one episode is {}/{}\n'.format(rewards_baseline.min(), rewards_baseline.max()))
            file.write('The Max/Min Steps is {}/{}'.format(steps_baseline.max(), steps_baseline.min()))
            file.close()

    # draw reward curves
    #plt.plot(rewards_baseline.mean(axis=0), label='BASELINE')
    #plt.xlabel('Episodes')
    #plt.ylabel('Sum of rewards during episode')
    #plt.ylim([-100, 0])
    #plt.legend()
    #plt.savefig('../images/figure_6_4_pg_baseline.png')
    #plt.close()

def cliffwalk_q():
    ''' This function runs Q-learning algorithm on cliffwalk '''
    # episodes of each run
    episodes = 5

    # perform 50 independent runs
    runs = 50

    agent_generator = lambda: QLearningAgent(alpha=ALPHA, gamma=GAMMA)

    rewards_q_learning = np.zeros((runs, episodes))

    for r in tqdm(range(runs)):
        rewards_q_learning[r,:] = trial(episodes, agent_generator)

    # draw reward curves
    plt.figure()
    sns.tsplot(data=rewards_q_learning, color='blue', condition='Q-Learning')

    #plt.plot(rewards_q_learning.mean(axis=0), label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    #plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('../images/figure_6_4_Q-Learning.png')
    plt.close()

def cliffwalk_random():
    ''' This function runs Q-learning algorithm on cliffwalk '''
    # episodes of each run
    episodes = 5

    # perform 50 independent runs
    runs = 50

    agent_generator = lambda: RandomAgent(alpha=ALPHA, gamma=GAMMA)

    rewards_random = np.zeros((runs, episodes))
    steps_random = np.zeros((runs, episodes))


    for r in tqdm(range(runs)):
        rewards_random[r,:], steps_random[r, :] = trial(episodes, agent_generator)

    # summarize rewards distribution and plot
    print('The Minimum Rewards/Steps is {}/{}'.format(rewards_random.min(), steps_random.min()))
    print('The Maximum Rewards/Steps is {}/{}'.format(rewards_random.max(), steps_random.max()))

    # draw reward curves
    plt.figure()
    sns.tsplot(data=rewards_random, color='blue', condition='Random')
    #plt.plot(rewards_random.mean(axis=0), label='Random')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    #plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('../images/figure_6_4_Random.png')
    plt.close()

# an episode with Sarsa
# @q_value: values for state action pair, will be updated
# @expected: if True, will use expected Sarsa algorithm
# @step_size: step size for updating
# @return: total rewards within this episode
def sarsa(q_value, expected=False, step_size=ALPHA):
    state = START
    action = choose_action(state, q_value)
    rewards = 0.0
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        rewards += reward
        if not expected:
            target = q_value[next_state[0], next_state[1], next_action]
        else:
            # calculate the expected value of new state
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    target += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    target += EPSILON / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]
        target *= GAMMA
        q_value[state[0], state[1], action] += step_size * (
                reward + target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return rewards

# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, step_size=ALPHA):
    state = START
    rewards = 0.0
    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        # Q-Learning update
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
    return rewards



# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)

# Use multiple runs instead of a single run and a sliding window
# With a single run I failed to present a smooth curve
# However the optimal policy converges well with a single run
# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def figure_6_4():
    # episodes of each run
    episodes = 500

    # perform 50 independent runs
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(0, episodes):
            # cut off the value by -100 to draw the figure more elegantly
            # rewards_sarsa[i] += max(sarsa(q_sarsa), -100)
            # rewards_q_learning[i] += max(q_learning(q_q_learning), -100)
            rewards_sarsa[i] += sarsa(q_sarsa)
            rewards_q_learning[i] += q_learning(q_q_learning)

    # averaging over independent runs
    rewards_sarsa /= runs
    rewards_q_learning /= runs

    # draw reward curves
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('../images/figure_6_4.png')
    plt.close()

    # display optimal policy
    print('Sarsa Optimal Policy:')
    print_optimal_policy(q_sarsa)
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)

# Due to limited capacity of calculation of my machine, I can't complete this experiment
# with 100,000 episodes and 50,000 runs to get the fully averaged performance
# However even I only play for 1,000 episodes and 10 runs, the curves looks still good.
def figure_6_6():
    step_sizes = np.arange(0.1, 1.1, 0.1)
    episodes = 1000
    runs = 10

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5
    methods = range(0, 6)

    performace = np.zeros((6, len(step_sizes)))
    for run in range(runs):
        for ind, step_size in tqdm(list(zip(range(0, len(step_sizes)), step_sizes))):
            q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
            q_expected_sarsa = np.copy(q_sarsa)
            q_q_learning = np.copy(q_sarsa)
            for ep in range(episodes):
                sarsa_reward = sarsa(q_sarsa, expected=False, step_size=step_size)
                expected_sarsa_reward = sarsa(q_expected_sarsa, expected=True, step_size=step_size)
                q_learning_reward = q_learning(q_q_learning, step_size=step_size)
                performace[ASY_SARSA, ind] += sarsa_reward
                performace[ASY_EXPECTED_SARSA, ind] += expected_sarsa_reward
                performace[ASY_QLEARNING, ind] += q_learning_reward

                if ep < 100:
                    performace[INT_SARSA, ind] += sarsa_reward
                    performace[INT_EXPECTED_SARSA, ind] += expected_sarsa_reward
                    performace[INT_QLEARNING, ind] += q_learning_reward

    performace[:3, :] /= episodes * runs
    performace[3:, :] /= 100 * runs
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']

    for method, label in zip(methods, labels):
        plt.plot(step_sizes, performace[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()

    plt.savefig('../images/figure_6_6.png')
    plt.close()

if __name__ == '__main__':
    #figure_6_4()
    #figure_6_6()
    cliffwalk_pg()
    #cliffwalk_q()
    #cliffwalk_random()
    #cliffwalk_pg_baseline()
