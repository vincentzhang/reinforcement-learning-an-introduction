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
import pandas as pd

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

# Whether to debug
DEBUG = False


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
    cutoff = 40000 # cutoff the episode at this number of steps
    flag_printed = False

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

            if num_steps % 20000 == 0:
                print("number of steps: {}".format(num_steps))
            if episode_end or num_steps >= cutoff:
                agent.episode_end(reward)
                if not episode_end:
                    print("Terminated at cutoff {}".format(cutoff))
                    if not flag_printed: # only print policy once at the first cutoff
                        agent.print_policy()
                        flag_printed = True
                #agent.store_history()
                break
            else:
                agent.update(next_state, reward)
            state = next_state # update the current state for next step

        rewards[episode_idx] = rewards_sum
        steps[episode_idx] = num_steps

        print("Reward/Step at Episode {} is {}/{}".format(episode_idx, rewards_sum, num_steps))

    agent.print_policy()
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
        self.step = 0

    def init_param(self, num_states, num_actions):
        self.theta = np.zeros((1, num_states * num_actions))
        self.x = np.zeros((self.theta.shape[1], num_actions))
        self.num_actions = num_actions
        self.num_states = num_states
        ## for visualization
        self.history_theta = []
        self.history_grad_ln_pi = []
        self.history_pmf = []
        self.history_rewards = []
        self.history_actions = []
        self.history_states = []
        self.history_G = []

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

    def get_pi_bk(self):
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
        #imax = np.argmax(pmf)

        if pmf[0, imin] < epsilon:
            import pdb;pdb.set_trace()
            #pmf[:] = (1 - epsilon)/(self.num_actions-1)
            # print("epsilone is ", epsilon)
            # print("pmf[:] is ", pmf[:])
            #raise Exception('x')

            #set min to epsilon
            # the other actions follow the original distribution of their values
            total = 1.0 - pmf[0, imin]
            pmf[0, imin] = epsilon

            # renormalize
            for j in range(pmf.shape[-1]):
                if j != imin:
                    pmf[0, j] = pmf[0, j] * (1-epsilon) / total

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

    def print_policy(self):
        """ print the current policy """
        policy = []
        print("the current policy is: ")
        for i in range(0, WORLD_HEIGHT):
            policy.append([])
            for j in range(0, WORLD_WIDTH):
                if [i, j] == GOAL:
                    policy[-1].append('G')
                    continue

                state_idx = self.state_to_idx([i, j])
                x = np.zeros_like(self.x)
                for action_idx in range(self.num_actions):
                    x[state_idx + self.num_states * action_idx, action_idx] = 1.0
                h = np.dot(self.theta, x) # theta: 1xSA, self.x: SAxA
                pmf = softmax(h)
                action = np.argmax(pmf)
                max_prob = pmf[0, action]
                if action == ACTION_UP:
                    policy[-1].append('U-{:.2f}'.format(max_prob))
                elif action == ACTION_DOWN:
                    policy[-1].append('D-{:.2f}'.format(max_prob))
                elif action == ACTION_LEFT:
                    policy[-1].append('L-{:.2f}'.format(max_prob))
                elif action == ACTION_RIGHT:
                    policy[-1].append('R-{:.2f}'.format(max_prob))
        for row in policy:
            print(row)

    def choose_action(self, state, reward):
        ''' Pick actions based on policy params '''
        self.step += 1
        #if self.step
        if reward is not None:
            self.rewards.append(reward)

        self.states.append(state)
        # first update x based on the state
        self.update_x(state)
        action = np.random.choice(ACTIONS, p=self.get_pi()[0, :])
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

            # store the history of theta for visualization later
            self.history_theta.append(self.theta.copy())
            self.history_grad_ln_pi.append(grad_ln_pi)
            self.history_pmf.append(pmf)

        self.history_rewards.append(self.rewards.copy())
        self.history_actions.append(self.actions.copy())
        self.history_states.append(self.states.copy())
        self.history_G.append(G.copy())

        #self.store_history()

        self.rewards = []
        self.actions = []
        self.states = []
        self.step = 0
        # self.history_theta = []
        # self.history_grad_ln_pi = []
        # self.history_pmf = []
        # self.history_rewards = []
        # self.history_actions = []
        # self.history_states = []

    def store_history(self):
        with open('log/history.npz', 'wb') as f:
            np.savez(f, theta=self.history_theta, grad=self.history_grad_ln_pi, pmf=self.history_pmf,
                     rewards = self.history_rewards, actions=self.history_actions, states=self.history_states,
                     G = self.history_G)
        print("History stored")


class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        super(ReinforceBaselineAgent, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w

    def init_param(self, num_states, num_actions):
        """ TODO: need to change the variables into the proper vector shape,
        for instance, Sx1 instead of 1xS"""
        self.theta = np.zeros((1, num_states * num_actions))
        self.x = np.zeros((self.theta.shape[1], num_actions))
        self.num_actions = num_actions
        self.num_states = num_states
        # w should be state-dependent, 48 states
        self.w = np.zeros((1, self.num_states)) # v(St, w)

    def store_history(self):
        pass

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
        Random Agent that performs equiprobable actions
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
        """ callback for performing update based on the next state,reward pair """
        pass

    def choose_action(self, state, reward):
        self.state = state
        action = np.random.choice(ACTIONS)
        self.actions.append(action) # need to memorize all the actions in this episode
        return action

    def episode_end(self, last_reward):
        self.actions = []
        self.state = None

    def print_policy(self):
        """ print the current policy """
        pass

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

class MCAgent(object):
    """
    Monte Carlo Agent that does on-policy every-visit MC control with epsilon-soft policies
    """
    def __init__(self, alpha, gamma, first_visit=False, num_states=None, num_actions=None):

        self.alpha = alpha
        self.gamma = gamma
        self.q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        self.num_actions = num_actions
        self.num_states = num_states

        self.actions = []
        self.rewards = []
        self.state = None
        self.states = []
        #self.states_indices = []
        self.states_visited = np.zeros(WORLD_HEIGHT*WORLD_WIDTH, dtype=bool)  # state vector with S items
        self.first_visit = first_visit
        # count the number to calculate moving average
        self.returns_counter = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))

    def init_param(self, num_states, num_actions):
        pass

    def update(self, next_state, reward):
        pass

    def state_to_idx(self, state):
        i, j = state
        state_idx = i * WORLD_WIDTH + j
        return state_idx

    def choose_action(self, state, reward):
        if reward is not None:
            self.rewards.append(reward)

        self.state = state
        self.states.append(state)

        action = choose_action(state, self.q_values)
        self.actions.append(action) # need to memorize all the actions in this episode
        return action

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        # a more efficient implementation, adopted in Chapter 5
        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        #gamma_pow = 1

        for i in range(len(G)):
            action = self.actions[i]
            state = self.states[i]
            state_idx = self.state_to_idx(state)

            if self.first_visit:
                # check if the state visited the first time in this episode
                if self.states_visited[state_idx]:
                    # already visited, no update
                    continue
                else:
                    # flag this state as visited
                    self.states_visited[state_idx] = True
            # update
            self.returns_counter[state[0],state[1],action] += 1

            #New = old * (n - 1) / n + newvalue / n
            n = self.returns_counter[state[0],state[1],action]
            # update action value
            self.q_values[state[0],state[1],action] = self.q_values[state[0],state[1],action]*(n -1)/n+G[i] / n

            #gamma_pow *= self.gamma # does not seem to be used

        self.rewards = []
        self.actions = []
        self.states = []
        self.state = None
        self.states_visited = np.zeros(WORLD_HEIGHT * WORLD_WIDTH, dtype=bool)

    def print_policy(self):
        """ print the current policy """
        print_optimal_policy(self.q_values)

class ActorCriticAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w, entropy_beta=0):
        super(ActorCriticAgent, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w
        self.gamma_pow = 1
        self.state = [0,0]
        self.step = 0
        self.beta = entropy_beta

    def init_param(self, num_states, num_actions):
        self.theta = np.zeros((1, num_states * num_actions))
        self.x = np.zeros((self.theta.shape[1], num_actions))
        self.num_actions = num_actions
        self.num_states = num_states
        # w should be state-dependent, 48 states
        self.w = np.zeros((1, self.num_states)) # v(St, w)

    def choose_action(self, state, reward):
        if reward is not None:
            # this reward is from the last iteration
            self.rewards.append(reward)
            self.state = state
            self.states.append(state)

        self.update_x(state)
        try:
            action = np.random.choice(ACTIONS, p=self.get_pi()[0, :])
        except ValueError as e:
            raise(e)

        self.actions.append(action)
        return action

    #def get_pi(self):
    #    h = np.dot(self.theta, self.x) # theta: 1xSA, self.x: SAxA
    #    pmf = softmax(h)
    #    return pmf

    def update(self, next_state, reward):
        self.step += 1
        next_state_idx = self.state_to_idx(next_state)
        state_idx = self.state_to_idx(self.state)
        delta = reward + self.gamma * self.w[0, next_state_idx] - self.w[0, state_idx]

        tmp = self.w[0, state_idx]
        self.w[0, state_idx] += self.alpha_w * delta # only update the weight for this state

        j = self.actions[-1] # x has already been updated when choosing the action for self.state
        pmf = self.get_pi()  # 1xA
        grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf[0, :])
        if self.beta > 0.0:
            update = self.alpha * (self.gamma_pow * delta * grad_ln_pi - self.beta * np.log(pmf[0, j]))
        else:
            update = self.alpha * self.gamma_pow * delta * grad_ln_pi

        self.theta += update
        self.gamma_pow *= self.gamma

        if DEBUG:
            print("alpha, alpha_w are {}, {}".format(self.alpha, self.alpha_w))
            print("Delta at step {} is {}".format(self.step, delta))
            print("w({})={}, w({})={}".format(state_idx, tmp,
                next_state_idx, self.w[0, next_state_idx]))
            print("the pi([{}, {}])={}".format(self.state[0], self.state[1], pmf))
            print("grad_ln_pi is {}".format(grad_ln_pi[grad_ln_pi!=0]))
            print("update is {}".format(update[update!=0]))
            print("theta max/min/mean: {}/{}/{}".format(self.theta.max(),
                self.theta.min(), self.theta.mean() ) )

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # perform the last update
        state_idx = self.state_to_idx(self.state)
        delta = last_reward - self.w[0, state_idx] # v(terminal)=0

        self.w[0, state_idx] += self.alpha_w * delta # only update the weight for this state

        j = self.actions[-1] # x has already been updated when choosing the action for self.state
        pmf = self.get_pi()  # 1xA
        grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf[0, :])
        if self.beta > 0.0:
            update = self.alpha * (self.gamma_pow * delta * grad_ln_pi - self.beta * np.log(pmf[0, j]))
        else:
            update = self.alpha * self.gamma_pow * delta * grad_ln_pi

        self.theta += update

        self.rewards = []
        self.actions = []
        self.states = []
        self.gamma_pow = 1
        self.step = 0


def cliffwalk_pg():
    ''' This function runs the REINFORCE algorithm on cliffwalk '''
    # episodes of each run
    episodes = 500

    # perform 50 independent runs
    runs = 50

    # settings of the REINFORCE agent
    alphas = [2**-28]#[2**-34, 2**-36]#[2**-20, 2**-22, 2**-24, 2**-26, 2**-28, 2**-30, 2**-32]
    #alphas = [2**-20]

    hyperparamsearch = False
    plot = False

    if not hyperparamsearch:
        np.random.seed(1973)

    global EPSILON
    EPSILON = 0

    for alpha in alphas:
        rewards_rei = np.zeros((runs, episodes))
        steps_rei = np.zeros((runs, episodes))
        agent_generator = lambda: ReinforceAgent(alpha=alpha, gamma=GAMMA)

        for r in tqdm(range(runs)):
            print("alpha: {}".format(alpha))
            rewards_rei[r, :], steps_rei[r, :] = trial(episodes, agent_generator)

        # stats of rewards, write to a txt file
        sum_rewards = rewards_rei.sum(axis=1).mean()  # this is the sum over episodes, averaged of the runs
        print("alpha: {}".format(alpha))
        print("sum of rewards: {}".format(sum_rewards))
        if hyperparamsearch:
            file = open("log/rei_sum_rewards_alpha_{}_rewards_{}.txt".format(alpha, sum_rewards), "w")
            file.write("sum of rewards mean: {} \n".format(sum_rewards))
            file.write('sum of rewards std: {} \n'.format(rewards_rei.sum(axis=1).std()))
            file.write('The Min/Max/Mean Reward in one episode is {}/{}/{}\n'.format(rewards_rei.min(), rewards_rei.max(), rewards_rei.mean()))
            file.write('The Max/Min/Mean Steps is {}/{}/{}'.format(steps_rei.max(), steps_rei.min(), steps_rei.mean()))
            file.close()
        else:
            # save the result
            with open('log/rewards_rei_alpha_{}_ep{}.npz'.format(alpha, episodes), 'wb') as data_f:
                np.savez(data_f, rewards=rewards_rei, steps=steps_rei)

            if plot:
                # draw reward curves
                plt.figure()
                sns.tsplot(data=rewards_rei, color='blue', condition='REINFORCE')
                #plt.plot(rewards_rei.mean(axis=0), label='REINFORCE')
                plt.xlabel('Episodes')
                plt.ylabel('Sum of rewards during episode')
                #plt.ylim([-100, 0])
                plt.legend()
                #plt.savefig('../images/figure_6_4_pg.png')
                plt.savefig('../images/figure_6_rei.png')
                plt.close()

def cliffwalk_pg_baseline():
    ''' This function runs the REINFORCE-baseline algorithm on cliffwalk '''
    # episodes of each run
    episodes = 100

    # perform 50 independent runs
    runs = 50

    hyperparamsearch = True
    plot = False

    if not hyperparamsearch:
        np.random.seed(1973)

    # settings of the REINFORCE agent
    alphas = [2**-16, 2**-18] #[2**-20, 2**-22]#2**-10, 2**-12, 2**-14,
    #alphas = [2**-20]
    alpha_ws = [2**-2] # 0.1/4 = 0.025
    #alpha_ws = [2**-6]

    global EPSILON
    EPSILON = 0

    for alpha in alphas:
        for alpha_w in alpha_ws:
            rewards_baseline = np.zeros((runs, episodes))
            steps_baseline = np.zeros((runs, episodes))

            agent_generator = lambda: ReinforceBaselineAgent(alpha=alpha, gamma=GAMMA, alpha_w=alpha_w)

            for r in tqdm(range(runs)):
                print("alpha: {}; alpha_w: {}".format(alpha, alpha_w))
                rewards_baseline[r, :], steps_baseline[r, :] = trial(episodes, agent_generator)

            # stats of rewards, write to a txt file
            sum_rewards = rewards_baseline.sum(axis=1).mean() # this is the sum over 200 episodes, averaged of 50 runs
            print("alpha: {}; alpha_w: {}".format(alpha, alpha_w))
            print("sum of rewards: {}".format(sum_rewards))
            #import pdb; pdb.set_trace()
            if hyperparamsearch:
                file = open("log/baseline_sum_rewards_alpha_{}_alphaw_{}_ep_{}_rewards_{}.txt".format(alpha, alpha_w, episodes, sum_rewards), "w")
                file.write("sum of rewards mean: {} \n".format(sum_rewards))
                file.write('sum of rewards std: {} \n'.format(rewards_baseline.sum(axis=1).std()))
                file.write(
                    'The Min/Max/Mean Reward in one episode is {}/{}/{}\n'.format(rewards_baseline.min(), rewards_baseline.max(),
                                                                                  rewards_baseline.mean()))
                file.write('The Max/Min/Mean Steps is {}/{}/{}'.format(steps_baseline.max(), steps_baseline.min(), steps_baseline.mean()))
                file.close()
            else:
                if plot:
                    # draw reward curves
                    plt.figure()
                    sns.tsplot(data=rewards_baseline, color='blue', condition='BASELINE')
                    #plt.plot(rewards_baseline.mean(axis=0), label='BASELINE')
                    plt.xlabel('Episodes')
                    plt.ylabel('Sum of rewards during episode')
                    #plt.ylim([-100, 0])
                    plt.legend()
                    #plt.savefig('../images/figure_6_4_pg_baseline_ep500.png')
                    plt.savefig('../images/figure_6_rei_baseline_alpha_{}_alphaw_{}_ep{}.png'.format(alpha, alpha_w, episodes))
                    plt.close()

def cliffwalk_pg_ac():
    ''' This function runs the actor-critic(ac_ algorithm on cliffwalk '''
    # episodes of each run
    episodes = 100

    # perform 50 independent runs
    runs = 50

    hyperparamsearch = True
    if not hyperparamsearch:
        np.random.seed(1973)

    # settings of the Actor Critic agent
    #alphas = [2**-8, 2**-10, 2**-12, 2**-14]#[2**-16, 2**-18, 2**-20, 2**-22]#2**-10, 2**-12, 2**-14,
    alphas = [2**-3]#[2**-16, 2**-18, 2**-20, 2**-22]#2**-10, 2**-12, 2**-14,
    #alphas = [2**-20]
    alpha_ws = [2**-1] #[2**-4, 2**-5, 2**-6] # 0.1/4 = 0.025
    #alpha_ws = [2**-6]

    global EPSILON
    EPSILON = 0

    for alpha in alphas:
        for alpha_w in alpha_ws:
            rewards_ac = np.zeros((runs, episodes))
            steps_ac = np.zeros((runs, episodes))

            agent_generator = lambda: ActorCriticAgent(alpha=alpha, gamma=GAMMA, alpha_w=alpha_w)

            for r in tqdm(range(runs)):
                print("alpha: {}; alpha_w: {}".format(alpha, alpha_w))
                rewards_ac[r, :], steps_ac[r, :] = trial(episodes, agent_generator)

            # stats of rewards, write to a txt file
            sum_rewards = rewards_ac.sum(axis=1).mean() # this is the sum over 200 episodes, averaged of 50 runs
            print("alpha: {}; alpha_w: {}".format(alpha, alpha_w))
            print("sum of rewards: {}".format(sum_rewards))
            if hyperparamsearch:
                file = open("log/ac0_sum_rewards_eps_{}_alpha_{}_alphaw_{}_rewards_{}.txt".format(EPSILON, alpha, alpha_w, sum_rewards), "w")
                file.write("sum of rewards: {} \n".format(sum_rewards))
                file.write('The Min/Max Reward in one episode is {}/{}\n'.format(rewards_ac.min(), rewards_ac.max()))
                file.write('The Max/Min Steps is {}/{}'.format(steps_ac.max(), steps_ac.min()))
                file.close()

    # draw reward curves
    if not hyperparamsearch:
        #with open('log/rewards_ac.npz', 'wb') as ac_f:
        #    np.savez(ac_f, ac=rewards_ac)
        plt.figure()
        sns.tsplot(data=rewards_ac, color='green', condition='Actor-Critic')
        f = np.load('log/rewards_q_sarsa.npz')
        rewards_q = f['q']
        rewards_sarsa = f['sarsa']
        sns.tsplot(data=rewards_q, color='red', condition='Q-learning')
        sns.tsplot(data=rewards_sarsa, color='blue', condition='Sarsa')
        #plt.plot(rewards_baseline.mean(axis=0), label='BASELINE')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        plt.ylim([-100, 0])
        plt.legend()
        plt.savefig('../images/figure_6_4_pg_all_ep500.png')
        plt.close()

def cliffwalk_pg_ac_test():
    ''' This function runs the actor-critic(ac_ algorithm on cliffwalk '''
    # episodes of each run
    episodes = 500

    # perform 50 independent runs
    runs = 50

    hyperparamsearch = False
    plot = False#True

    if not hyperparamsearch:
        np.random.seed(1973)

    # settings of the Actor Critic agent
    #alphas = [2**-8, 2**-10, 2**-12, 2**-14]#[2**-16, 2**-18, 2**-20, 2**-22]#2**-10, 2**-12, 2**-14,
    alphas = [2**-4]#[2**-6]#[2**-3]#[2**-6]#[2**-16, 2**-18, 2**-20, 2**-22]#2**-10, 2**-12, 2**-14,
    #alphas = [2**-20]
    alpha_ws = [2**-4]#[2**-1]#[2**-2]#[0.5] #[2**-4, 2**-5, 2**-6] # 0.1/4 = 0.025
    #alpha_ws = [2**-6]

    global EPSILON
    EPSILON = 0

    for alpha in alphas:
        for alpha_w in alpha_ws:
            rewards_ac = np.zeros((runs, episodes))
            steps_ac = np.zeros((runs, episodes))

            agent_generator = lambda: ActorCriticAgent(alpha=alpha, gamma=GAMMA, alpha_w=alpha_w)
            for r in tqdm(range(runs)):
                print("alpha: {}; alpha_w: {}; EPSILON: {}".format(alpha, alpha_w, EPSILON))
                rewards_ac[r, :], steps_ac[r, :] = trial(episodes, agent_generator)

            # stats of rewards, write to a txt file
            sum_rewards = rewards_ac.sum(axis=1).mean() # this is the sum over 200 episodes, averaged of 50 runs
            print("alpha: {}; alpha_w: {}; EPSILON: {}".format(alpha, alpha_w, EPSILON))
            print("sum of rewards: {}".format(sum_rewards))
            if hyperparamsearch:
                file = open("log/ac0_sum_rewards_alpha_{}_alphaw_{}_rewards_{}.txt".format(alpha, alpha_w, sum_rewards), "w")
                file.write("sum of rewards: {} \n".format(sum_rewards))
                file.write('The Min/Max Reward in one episode is {}/{}\n'.format(rewards_ac.min(), rewards_ac.max()))
                file.write('The Max/Min Steps is {}/{}'.format(steps_ac.max(), steps_ac.min()))
                file.close()
            # draw reward curves
            else:
                with open('log/rewards_ac_eps_{}_alpha_{}_alphaw_{}_ep{}.npz'.format(EPSILON, alpha, alpha_w, episodes), 'wb') as ac_f:
                    np.savez(ac_f, ac=rewards_ac)
                print('The Min/Max Reward in one episode is {}/{}\n'.format(rewards_ac.min(), rewards_ac.max()))
                print('The Max/Min Steps is {}/{}'.format(steps_ac.max(), steps_ac.min()))

            if plot:
                plt.figure()
                smoothing_window = 20
                rewards_ac_smoothed = pd.Series(rewards_ac.mean(axis=0)).rolling(smoothing_window,
                                                                                min_periods=smoothing_window).mean()
                plt.plot(rewards_ac_smoothed, label='Actor-Critic', color='green')
                #sns.tsplot(data=rewards_ac, color='green', condition='Actor-Critic')
                f = np.load('log/rewards_q_sarsa_eps_0.1.npz')
                rewards_q = f['q']
                rewards_sarsa = f['sarsa']
                rewards_q_smoothed = pd.Series(rewards_q.mean(axis=0)).rolling(smoothing_window,
                                                                               min_periods=smoothing_window).mean()
                rewards_sarsa_smoothed = pd.Series(rewards_sarsa.mean(axis=0)).rolling(smoothing_window,
                                                                                       min_periods=smoothing_window).mean()
                plt.plot(rewards_q_smoothed, label='Q-learning', color='red')
                plt.plot(rewards_sarsa_smoothed, label='Sarsa', color='blue')
                #sns.tsplot(data=rewards_q, color='red', condition='Q-learning')
                #sns.tsplot(data=rewards_sarsa, color='blue', condition='Sarsa')
                #plt.plot(rewards_baseline.mean(axis=0), label='BASELINE')
                plt.xlabel('Episodes')
                plt.ylabel('Sum of rewards during episode')
                plt.ylim([-100, 0])
                plt.yticks(np.arange(-100, 0.01, step=10))
                plt.legend()
                # Then extract the spines and make them invisible
                plt.gca().spines['right'].set_color('none')
                plt.gca().spines['top'].set_color('none')
                plt.savefig('../images/figure_6_4_pg_all_eps_{}_alpha_{}_alphaw_{}_ep500_test.png'.format(EPSILON, alpha, alpha_w))
                #plt.savefig('../images/figure_6_4_pg_ac_eps_{}_ep2000.png'.format(EPSILON))
                #plt.savefig('../images/figure_6_4_pg_all_eps_{}_ep500.png'.format(EPSILON))
                plt.close()


def cliffwalk_ac_ent():
    ''' This function runs the actor-critic (with entropy regularization) algorithm on cliffwalk '''
    # episodes of each run
    episodes = 100

    # perform 50 independent runs
    runs = 50

    hyperparamsearch = True
    plot = False

    if not hyperparamsearch:
        np.random.seed(1973)

    # settings of the Actor Critic agent
    #alphas = [2**-8, 2**-10, 2**-12, 2**-14]#[2**-16, 2**-18, 2**-20, 2**-22]#2**-10, 2**-12, 2**-14,
    alphas = [2**-4]#, 2**-6, 2**-8]#[2**-16, 2**-18, 2**-20, 2**-22]#2**-10, 2**-12, 2**-14,
    #alphas = [2**-20]
    alpha_ws = [2**-4]#, 2**-6] #[2**-4, 2**-5, 2**-6] # 0.1/4 = 0.025
    #alpha_ws = [2**-6]
    betas = [2**-3, 2**-4]


    for alpha in alphas:
        for alpha_w in alpha_ws:
            for beta in betas:
                rewards_ac = np.zeros((runs, episodes))
                steps_ac = np.zeros((runs, episodes))

                agent_generator = lambda: ActorCriticAgent(alpha=alpha, gamma=GAMMA, alpha_w=alpha_w, entropy_beta=beta)
                for r in tqdm(range(runs)):
                    print("alpha: {}; alpha_w: {}; entropy_beta: {}".format(alpha, alpha_w, beta))
                    rewards_ac[r, :], steps_ac[r, :] = trial(episodes, agent_generator)

                # stats of rewards, write to a txt file
                sum_rewards = rewards_ac.sum(axis=1).mean() # this is the sum over 200 episodes, averaged of 50 runs
                print("alpha: {}; alpha_w: {}; entropy_beta: {}".format(alpha, alpha_w, beta))
                print("sum of rewards: {}".format(sum_rewards))
                if hyperparamsearch:
                    file = open("log/ac_ent_sum_rewards_alpha_{}_alphaw_{}_beta_{}_rewards_{}.txt".format(alpha,
                        alpha_w, beta, sum_rewards), "w")
                    file.write("sum of rewards: {} \n".format(sum_rewards))
                    file.write('The Min/Max Reward in one episode is {}/{}\n'.format(rewards_ac.min(), rewards_ac.max()))
                    file.write('The Max/Min Steps is {}/{}'.format(steps_ac.max(), steps_ac.min()))
                    file.close()
                else:
                    # draw reward curves
                    with open('log/rewards_ac_eps_{}_alpha_{}_alphaw_{}_beta_{}_ep{}.npz'.format(EPSILON, alpha, alpha_w, beta, episodes), 'wb') as ac_f:
                        np.savez(ac_f, ac=rewards_ac)
                if plot:
                    plt.figure()
                    #sns.tsplot(data=rewards_ac, color='green', condition='Actor-Critic')
                    smoothing_window = 20
                    rewards_ac_smoothed = pd.Series(rewards_ac.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
                    plt.plot(rewards_ac_smoothed, label='Actor-Critic', color='green')
                    #plt.text(450,rewards_ac_smoothed.at[499],'actor-critic')
                    f = np.load('log/rewards_q_sarsa_eps_0.1.npz')
                    rewards_q = f['q']
                    rewards_sarsa = f['sarsa']
                    rewards_q_smoothed = pd.Series(rewards_q.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
                    rewards_sarsa_smoothed = pd.Series(rewards_sarsa.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
                    plt.plot(rewards_q_smoothed, label='Q-learning', color='red')
                    plt.plot(rewards_sarsa_smoothed, label='Sarsa', color='blue')
                    #sns.tsplot(data=rewards_q, color='red', condition='Q-learning')
                    #sns.tsplot(data=rewards_sarsa, color='blue', condition='Sarsa')
                    plt.xlabel('Episodes')
                    plt.ylabel('Sum of rewards during episode')
                    plt.ylim([-100, 0])
                    plt.yticks(np.arange(-100, 0.01, step=10))
                    plt.legend()
                    # remove all the ticks and directly label each bar with respective value
                    #plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
                    #plt.box(False)
                    # Then extract the spines and make them invisible
                    plt.gca().spines['right'].set_color('none')
                    plt.gca().spines['top'].set_color('none')
                    #plt.grid(True)
                    plt.savefig('../images/figure_6_all_eps_{}_alpha_{}_alphaw_{}_beta_{}_ep500.png'.format(EPSILON, alpha, alpha_w, beta))
                    #plt.savefig('../images/figure_6_4_pg_ac_eps_{}_ep2000.png'.format(EPSILON))
                    #plt.savefig('../images/figure_6_4_pg_all_eps_{}_ep500.png'.format(EPSILON))
                    plt.close()

def make_figure():
    eps_list = [0.1, 0.05, 0.01]
    for eps in eps_list:
        plt.figure()
        f = np.load('log/rewards_ac_eps_{}.npz'.format(eps))
        rewards_ac = f['ac']
        f.close()
        sns.tsplot(data=rewards_ac, color='green', condition='Actor-Critic')
        f = np.load('log/rewards_q_sarsa_eps_{}.npz'.format(eps))
        rewards_q = f['q']
        rewards_sarsa = f['sarsa']
        f.close()
        sns.tsplot(data=rewards_q, color='red', condition='Q-learning')
        sns.tsplot(data=rewards_sarsa, color='blue', condition='Sarsa')
        # plt.plot(rewards_baseline.mean(axis=0), label='BASELINE')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        plt.ylim([-100, 0])
        plt.legend()
        plt.savefig('../images/figure_6_4_pg_all_eps_{}_ep500.png'.format(eps))
        plt.close()

def compare_all():
    alpha = 0.9
    f = np.load('log/rewards_q_sarsa_eps_0.1_alpha_{}.npz'.format(alpha))
    rewards_q = f['q']
    f.close()
    #
    f = np.load('log/rewards_q_sarsa_eps_0.1.npz')
    rewards_sarsa = f['sarsa']
    f.close()
    
    f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep500.npz')
    rewards_ac_orig = f['ac']
    f.close()
    
    plt.figure()
    smoothing_window = 20
    rewards_ac_orig_smoothed = pd.Series(rewards_ac_orig.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic Original', color='green')
    rewards_q_smoothed = pd.Series(rewards_q.mean(axis=0)).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
    rewards_sarsa_smoothed = pd.Series(rewards_sarsa.mean(axis=0)).rolling(smoothing_window,
                                                                           min_periods=smoothing_window).mean()
    plt.plot(rewards_q_smoothed, label='Q-learning', color='red')
    plt.plot(rewards_sarsa_smoothed, label='Sarsa', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.yticks(np.arange(-100, 0.01, step=10))
    plt.legend()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.savefig('../images/figure_6_all_ep500_q_{}.png'.format(alpha))
    plt.close()


def compare_ac_ent():
    #f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep500.npz')
    f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep1000.npz') # last
    #f = np.load('log/rewards_ac_eps_0_alpha_0.0625_alphaw_0.0625_ep1000.npz')
    rewards_ac_orig = f['ac']
    f.close()
    #f = np.load('log/rewards_ac_eps_0.1.npz')
    f = np.load('log/rewards_ac_eps_0.1_alpha_0.015625_alphaw_0.5_ep1000.npz')
    rewards_ac_eps = f['ac']
    f.close()
    f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.125_ep1000.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.125_ep500.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.5_ep500.npz') # best?
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.125_alphaw_0.0625_beta_0.0625_ep500.npz') # one of the suboptimal
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.5_beta_0.5_ep500.npz') # early convg
    rewards_ac_ent = f['ac']
    f.close()

    plt.figure()
    smoothing_window = 20
    rewards_ac_orig_smoothed = pd.Series(rewards_ac_orig.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic Original', color='green')
    rewards_ac_eps_smoothed = pd.Series(rewards_ac_eps.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_eps_smoothed, label='Actor-Critic soft eps', color='blue')

    rewards_ac_ent_smoothed = pd.Series(rewards_ac_ent.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_ent_smoothed, label='Actor-Critic entropy regularization', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.yticks(np.arange(-100, 0.01, step=10))
    plt.legend()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.savefig('../images/figure_6_all_ac_ep1000_orig.png')
    plt.close()


def compare_ac_ent2():
    #f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep500.npz')
    f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep1000.npz') # last
    rewards_ac_orig = f['ac']
    f.close()
    f = np.load('log/rewards_ac_eps_0_alpha_0.0625_alphaw_0.0625_ep1000.npz')
    #f = np.load('log/rewards_ac_eps_0.1.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.015625_alphaw_0.5_ep1000.npz')
    rewards_ac_eps = f['ac']
    f.close()
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.125_ep1000.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.125_ep500.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.5_ep500.npz') # best?
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.125_alphaw_0.0625_beta_0.0625_ep500.npz') # one of the suboptimal
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.5_beta_0.5_ep500.npz') # early convg
    #rewards_ac_ent = f['ac']
    #f.close()

    plt.figure()
    smoothing_window = 20
    rewards_ac_orig_smoothed = pd.Series(rewards_ac_orig.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic With Best Interim Performance', color='green')
    rewards_ac_eps_smoothed = pd.Series(rewards_ac_eps.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_eps_smoothed, label='Actor-Critic With Sub-Optimal Interm Performance', color='blue')

    #rewards_ac_ent_smoothed = pd.Series(rewards_ac_ent.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #plt.plot(rewards_ac_ent_smoothed, label='Actor-Critic entropy regularization', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.yticks(np.arange(-100, 0.01, step=10))
    plt.legend()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.savefig('../images/figure_6_all_ac_ep1000_convergence.png')
    plt.close()


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

def cliffwalk_mc():
    ''' This function runs MC algorithm on cliffwalk '''
    # episodes of each run
    episodes = 100

    # perform 50 independent runs
    runs = 50

    plot = True

    np.random.seed(1973)

    agent_generator = lambda: MCAgent(alpha=ALPHA, gamma=GAMMA)

    rewards_mc_learning = np.zeros((runs, episodes))
    steps_mc_learning = np.zeros((runs, episodes))

    for r in tqdm(range(runs)):
        rewards_mc_learning[r,:], steps_mc_learning[r, :] = trial(episodes, agent_generator)

    # summarize rewards distribution and plot
    sum_rewards = rewards_mc_learning.sum(axis=1).mean()  # this is the sum over episodes, averaged of the runs
    print("sum of rewards mean: {}".format(sum_rewards))
    print('The Minimum Rewards/Steps is {}/{}'.format(rewards_mc_learning.min(), steps_mc_learning.min()))
    print('The Maximum Rewards/Steps is {}/{}'.format(rewards_mc_learning.max(), steps_mc_learning.max()))

    # write the result to file
    file = open("log/mc_sum_rewards_ep{}_rewards_{}.txt".format(episodes, sum_rewards), "w")
    file.write("sum of rewards mean: {} \n".format(sum_rewards))
    file.write('sum of rewards std: {} \n'.format(rewards_mc_learning.sum(axis=1).std()))
    file.write('The Min/Max/Mean Reward in one episode is {}/{}/{}\n'.format(rewards_mc_learning.min(), rewards_mc_learning.max(),
                                                                             rewards_mc_learning.mean()))
    file.write('The Max/Min/Mean Steps is {}/{}/{}'.format(steps_mc_learning.max(), steps_mc_learning.min(), steps_mc_learning.mean()))
    file.close()

    # write the rewards to a file
    with open('log/rewards_mc_ep{}.npz'.format(episodes), 'wb') as data_f:
        np.savez(data_f, mc=rewards_mc_learning, mc_steps=steps_mc_learning)

    if plot:
        # draw reward curves
        plt.figure()
        sns.tsplot(data=rewards_mc_learning, color='blue', condition='MC')
        #plt.plot(rewards_mc_learning.mean(axis=0), label='MC')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        #plt.ylim([-100, 0])
        plt.legend()
        #plt.savefig('../images/figure_6_4_MC.png')
        plt.savefig('../images/figure_6_MC_ep{}.png'.format(episodes))
        plt.close()

def cliffwalk_mc_first():
    ''' This function runs MC algorithm on cliffwalk '''
    # episodes of each run
    episodes = 100

    # perform 50 independent runs
    runs = 50

    plot = True

    np.random.seed(1973)

    agent_generator = lambda: MCAgent(alpha=ALPHA, gamma=GAMMA, first_visit=True)

    rewards_mc_learning = np.zeros((runs, episodes))
    steps_mc_learning = np.zeros((runs, episodes))

    for r in tqdm(range(runs)):
        rewards_mc_learning[r,:], steps_mc_learning[r, :] = trial(episodes, agent_generator)

    # summarize rewards distribution and plot
    sum_rewards = rewards_mc_learning.sum(axis=1).mean()  # this is the sum over episodes, averaged of the runs
    print("sum of rewards mean: {}".format(sum_rewards))
    print('The Minimum Rewards/Steps is {}/{}'.format(rewards_mc_learning.min(), steps_mc_learning.min()))
    print('The Maximum Rewards/Steps is {}/{}'.format(rewards_mc_learning.max(), steps_mc_learning.max()))

    # write the result to file
    file = open("log/mc_first_sum_rewards_ep{}_rewards_{}.txt".format(episodes, sum_rewards), "w")
    file.write("sum of rewards mean: {} \n".format(sum_rewards))
    file.write('sum of rewards std: {} \n'.format(rewards_mc_learning.sum(axis=1).std()))
    file.write('The Min/Max/Mean Reward in one episode is {}/{}/{}\n'.format(rewards_mc_learning.min(), rewards_mc_learning.max(),
                                                                             rewards_mc_learning.mean()))
    file.write('The Max/Min/Mean Steps is {}/{}/{}'.format(steps_mc_learning.max(), steps_mc_learning.min(), steps_mc_learning.mean()))
    file.close()

    # write the rewards to a file
    with open('log/rewards_mc_first_ep{}.npz'.format(episodes), 'wb') as data_f:
        np.savez(data_f, mc=rewards_mc_learning, mc_steps=steps_mc_learning)

    if plot:
        # draw reward curves
        plt.figure()
        sns.tsplot(data=rewards_mc_learning, color='blue', condition='MC first-visit')
        #plt.plot(rewards_mc_learning.mean(axis=0), label='MC')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        #plt.ylim([-100, 0])
        plt.legend()
        #plt.savefig('../images/figure_6_4_MC.png')
        plt.savefig('../images/figure_6_MC_first_ep{}.png'.format(episodes))
        plt.close()

def cliffwalk_random():
    ''' This function runs Q-learning algorithm on cliffwalk '''
    # episodes of each run
    episodes = 100

    # perform 50 independent runs
    runs = 50

    plot = True

    np.random.seed(1973)

    agent_generator = lambda: RandomAgent(alpha=ALPHA, gamma=GAMMA)

    rewards_random = np.zeros((runs, episodes))
    steps_random = np.zeros((runs, episodes))

    for r in tqdm(range(runs)):
        rewards_random[r,:], steps_random[r, :] = trial(episodes, agent_generator)

    # summarize rewards distribution and plot
    sum_rewards = rewards_random.sum(axis=1).mean()  # this is the sum over episodes, averaged of the runs
    print("sum of rewards mean: {}".format(sum_rewards))
    print('The Minimum Rewards/Steps is {}/{}'.format(rewards_random.min(), steps_random.min()))
    print('The Maximum Rewards/Steps is {}/{}'.format(rewards_random.max(), steps_random.max()))

    # write the result to file
    file = open("log/random_sum_rewards_ep{}_rewards_{}.txt".format(episodes, sum_rewards), "w")
    file.write("sum of rewards mean: {} \n".format(sum_rewards))
    file.write('sum of rewards std: {} \n'.format(rewards_random.sum(axis=1).std()))
    file.write('The Min/Max/Mean Reward in one episode is {}/{}/{}\n'.format(rewards_random.min(), rewards_random.max(),
                                                                             rewards_random.mean()))
    file.write('The Max/Min/Mean Steps is {}/{}/{}'.format(steps_random.max(), steps_random.min(), steps_random.mean()))
    file.close()

    # write the rewards to a file
    with open('log/rewards_random_ep{}.npz'.format(episodes), 'wb') as random_f:
        np.savez(random_f, random=rewards_random, random_steps=steps_random)

    if plot:
        # draw reward curves
        plt.figure()
        sns.tsplot(data=rewards_random, color='blue', condition='Random')
        #plt.plot(rewards_random.mean(axis=0), label='Random')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        #plt.ylim([-100, 0])
        plt.legend()
        plt.savefig('../images/figure_6_Random_ep{}.png'.format(episodes))
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

def figure_6_q_sarsa():
    np.random.seed(1973)

    # episodes of each run
    episodes = 500

    # perform 50 independent runs
    runs = 50

    rewards_sarsa = np.zeros((runs, episodes))
    rewards_q_learning = np.zeros((runs, episodes))
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(0, episodes):
            # cut off the value by -100 to draw the figure more elegantly
            # rewards_sarsa[i] += max(sarsa(q_sarsa), -100)
            # rewards_q_learning[i] += max(q_learning(q_q_learning), -100)
            rewards_sarsa[r, i] = sarsa(q_sarsa)
            rewards_q_learning[r, i] = q_learning(q_q_learning)

    # store the rewards into npz array
    with open('log/rewards_q_sarsa.npz', 'wb') as f:
        np.savez(f, q=rewards_q_learning, sarsa=rewards_sarsa)

    # averaging over independent runs
    # draw reward curves
    plt.plot(rewards_sarsa.mean(axis=0), label='Sarsa')
    plt.plot(rewards_q_learning.mean(axis=0), label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('../images/figure_6_q_sarsa.png')
    plt.close()


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

    #cliffwalk_mc()
    #cliffwalk_q()
    #cliffwalk_random()
    #cliffwalk_pg()
    cliffwalk_pg_baseline()
    #cliffwalk_pg_ac()
    #cliffwalk_pg_ac_test()
    #compare_all()
    #cliffwalk_ac_ent()
    #figure_6_q_sarsa()
    #make_figure()
    #compare_ac_ent()
    #compare_ac_ent2()
