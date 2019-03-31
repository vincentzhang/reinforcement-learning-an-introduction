import numpy as np
from matplotlib import pyplot as plt


npzf = np.load('log/history.npz')
keys = npzf.files
#['theta', 'grad', 'pmf', 'rewards', 'actions', 'states']

theta = npzf['theta']
####
# this is to plot 4 subfigures of the action parameter for the four actions at the starting state (3,0)
#fig = plt.figure()
fig, ax = plt.subplots(4, 1, sharex='col')
ax[0].plot(theta[:, 0, 36], label="Policy Parameter for state [3,0], UP")
ax[0].legend()
ax[0].set_ylabel('Theta')
ax[1].plot(theta[:, 0, 36+48], label="Policy Parameter for state [3,0], DOWN")
ax[1].legend()
ax[1].set_ylabel('Theta')
ax[2].plot(theta[:, 0, 36+48*2], label="Policy Parameter for state [3,0], LEFT")
ax[2].legend()
ax[2].set_ylabel('Theta')
ax[3].plot(theta[:, 0, 36+48*3], label="Policy Parameter for state [3,0], RIGHT")
ax[3].legend()
ax[3].set_ylabel('Theta')

plt.xlabel('Steps')
#plt.ylabel('Theta')
#plt.legend()
plt.savefig('../images/figure_6_pg_state_0.png')
plt.close()
####
