import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from tqdm import tqdm
#import seaborn as sns
import pandas as pd


def figure_mc_comparison():
    ''' Compare REINFORCE, BASELINE, MC first-visit, MC every-visit, Random '''
    # REINFORCE

    f = np.load('log/rewards_rei_alpha_3.725290298461914e-09_ep500.npz')
    rewards_rei = f['rewards']
    f.close()
    #
    f = np.load('log/rewards_baseline_alpha_3.814697265625e-06_alphaw_0.0625_ep500.npz')
    rewards_baseline = f['rewards']
    mask_row = np.ones(50, dtype=bool)
    mask_row[45] = False
    rewards_baseline = rewards_baseline[mask_row, :] # remove 45-th row since it's all 4000000
    f.close()
    # mc every-visit
    f = np.load('log/rewards_mc_ep500.npz')
    rewards_mc_every = f['mc']
    f.close()
    # mc first-visit
    f = np.load('log/rewards_mc_first_ep500.npz')
    rewards_mc_first = f['mc']
    f.close()
    # random
    f = np.load('log/rewards_random_ep500.npz')
    rewards_random = f['rewards']
    f.close()


    plt.figure()
    smoothing_window = 20
    rewards_rei_smoothed = pd.Series(rewards_rei.mean(axis=0)).rolling(smoothing_window,
                                                                               min_periods=smoothing_window).mean()

    rewards_baseline_smoothed = pd.Series(rewards_baseline.mean(axis=0)).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
    rewards_mc_every_smoothed = pd.Series(rewards_mc_every.mean(axis=0)).rolling(smoothing_window,
                                                                           min_periods=smoothing_window).mean()
    rewards_mc_first_smoothed = pd.Series(rewards_mc_first.mean(axis=0)).rolling(smoothing_window,
                                                                                 min_periods=smoothing_window).mean()
    rewards_random_smoothed = pd.Series(rewards_random.mean(axis=0)).rolling(smoothing_window,
                                                                                 min_periods=smoothing_window).mean()
    plt.plot(rewards_rei_smoothed, label='REINFORCE')
    plt.plot(rewards_baseline_smoothed, label='REINFORCE with baseline')
    plt.plot(rewards_mc_every_smoothed, label='MC every-visit')
    plt.plot(rewards_mc_first_smoothed, label='MC first-visit')
    plt.plot(rewards_random_smoothed, label='Random')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    #plt.ylim([-100, 0])
    #plt.yticks(np.arange(-100, 0.01, step=10))
    plt.legend()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.savefig('../images/figure_6_mc_comp_ep500.png')
    plt.close()


def compare_td():
    alpha = 0.9
    f = np.load('log/rewards_q_sarsa_eps_0.1_alpha_{}.npz'.format(alpha))
    rewards_q = f['q']
    f.close()
    #
    f = np.load('log/rewards_q_sarsa_eps_0.1.npz')
    rewards_sarsa = f['sarsa']
    f.close()

    f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep500.npz')
    rewards_ac_orig = f['rewards']
    f.close()

    plt.figure()
    smoothing_window = 20
    rewards_ac_orig_smoothed = pd.Series(rewards_ac_orig.mean(axis=0)).rolling(smoothing_window,
                                                                               min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic', color='green')
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
    #plt.savefig('../images/figure_6_all_ep500_q_{}.png'.format(alpha))
    plt.savefig('../images/figure_6_td_ep500_q_{}.png'.format(alpha))
    plt.close()

def compare_ac_ent():
    #f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep500.npz')
    #f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep1000.npz') # best interim
    f = np.load('log/rewards_ac_eps_0_alpha_0.0625_alphaw_0.0625_ep1000.npz') # best aym, no entropy
    rewards_ac_orig = f['ac']
    f.close()
    #f = np.load('log/rewards_ac_eps_0.1.npz')
    f = np.load('log/rewards_ac_eps_0.1_alpha_0.015625_alphaw_0.5_ep1000.npz') # soft eps 0.1
    rewards_ac_eps = f['ac']
    f.close()
    #f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_beta_0.125_ep1000.npz') # entropy regularization
    f = np.load('log/rewards_ac_eps_0_alpha_0.0625_alphaw_0.0625_beta_0.125_ep1000.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.125_ep500.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.5_ep500.npz') # best?
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.125_alphaw_0.0625_beta_0.0625_ep500.npz') # one of the suboptimal
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.5_beta_0.5_ep500.npz') # early convg
    #rewards_ac_ent = f['rewards']
    rewards_ac_ent = f['ac']
    f.close()

    plt.figure()
    smoothing_window = 20
    rewards_ac_orig_smoothed = pd.Series(rewards_ac_orig.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic Original',color='#8DBA43') # green
    rewards_ac_eps_smoothed = pd.Series(rewards_ac_eps.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_eps_smoothed, label='Actor-Critic Soft eps',
            color='#FF8111') #e24a33
    rewards_ac_ent_smoothed = pd.Series(rewards_ac_ent.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_ent_smoothed, label='Actor-Critic Entropy Regularization', color='#8479d1')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.yticks(np.arange(-100, 0.01, step=10))
    plt.legend()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.savefig('../images/figure_6_all_ac_ep1000_multicolor.png')
    plt.close()

def compare_ac_ent_v2():
    #f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep500.npz')
    f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep1000.npz') # best interim
    #f = np.load('log/rewards_ac_eps_0_alpha_0.0625_alphaw_0.0625_ep1000.npz') # best aym, no entropy
    #rewards_ac_orig = f['ac']
    rewards_ac_orig = f['rewards']
    f.close()
    #f = np.load('log/rewards_ac_eps_0.1.npz')
    f = np.load('log/rewards_ac_eps_0.1_alpha_0.015625_alphaw_0.5_ep1000.npz') # soft eps 0.1
    rewards_ac_eps = f['ac']
    f.close()
    #f = np.load('log/rewards_ac_v2_eps_0_alpha_0.125_alphaw_0.25_beta_0.5_ep1000.npz')  # entropy regularization, v2
    f = np.load('log/rewards_ac_v3_eps_0_alpha_0.125_alphaw_0.25_beta_0.125_ep1000.npz') # entropy regularization, v3
    #f = np.load('log/rewards_ac_eps_0_alpha_0.0625_alphaw_0.0625_beta_0.125_ep1000.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.125_ep500.npz')
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.0625_beta_0.5_ep500.npz') # best?
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.125_alphaw_0.0625_beta_0.0625_ep500.npz') # one of the suboptimal
    #f = np.load('log/rewards_ac_eps_0.1_alpha_0.0625_alphaw_0.5_beta_0.5_ep500.npz') # early convg
    rewards_ac_ent = f['rewards']
    #rewards_ac_ent = f['ac']
    f.close()

    plt.figure()
    smoothing_window = 1
    rewards_ac_orig_smoothed = pd.Series(rewards_ac_orig.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic Original',color='#8DBA43') # green
    rewards_ac_eps_smoothed = pd.Series(rewards_ac_eps.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_eps_smoothed, label='Actor-Critic Soft eps',
            color='#FF8111') #e24a33
    rewards_ac_ent_smoothed = pd.Series(rewards_ac_ent.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_ac_ent_smoothed, label='Actor-Critic Entropy Regularization', color='#8479d1')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.yticks(np.arange(-100, 0.01, step=10))
    plt.legend()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.savefig('../images/figure_6_all_ac_ep1000_multicolor.png')
    plt.close()

def compare_ac_learningcurve():
    #f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep500.npz')
    f = np.load('log/rewards_ac_eps_0_alpha_0.125_alphaw_0.25_ep1000.npz') # faster
    rewards_ac_orig = f['rewards']
    f.close()
    f = np.load('log/rewards_ac_eps_0_alpha_0.0625_alphaw_0.0625_ep1000.npz') # smoother
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
    #plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic With Best Interim Performance', color='green')
    plt.plot(rewards_ac_orig_smoothed, label='Actor-Critic With Best Interim Performance')
    rewards_ac_eps_smoothed = pd.Series(rewards_ac_eps.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #plt.plot(rewards_ac_eps_smoothed, label='Actor-Critic With Sub-Optimal Interm Performance', color='blue')
    plt.plot(rewards_ac_eps_smoothed, label='Actor-Critic With Sub-Optimal Interm Performance')

    #rewards_ac_ent_smoothed = pd.Series(rewards_ac_ent.mean(axis=0)).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #plt.plot(rewards_ac_ent_smoothed, label='Actor-Critic entropy regularization', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.yticks(np.arange(-100, 0.01, step=10))
    #plt.legend()
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    #plt.savefig('../images/figure_6_all_ac_ep1000_convergence.png')
    plt.savefig('../images/figure_6_ac_ep1000_learningcurve.png')
    plt.close()

#figure_mc_comparison()
#compare_ac_ent()
compare_ac_ent_v2()
#compare_ac_learningcurve()
#compare_td()
