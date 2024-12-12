import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d

#### symbols
theta = ('\\theta')
delta = ('\delta')


fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')

# X = difference between Adv(x*) and Adv(x) in standard deviations
X = np.linspace(0, 2, 101)
# Z = min-max scaled entropy
Z = np.linspace(0, 1, 101)
X, Z = np.meshgrid(X, Z)

# only delta x*
Y = (1-Z)*(-X/2) + (Z)*((1-X/2)**10-1)
ax3d.set_ylabel("b")

# only delta x
# Y = (1-Z)*(X/2 + 1) + (Z)*((1-X/2)**10)
# ax3d.set_ylabel("a")

ax3d.yaxis.label.set_rotation(0)
ax3d.yaxis.label.set_fontsize(15)
ax3d.yaxis.labelpad = 20

#setting axis titles
ax3d.set_xlabel('A(x*) - A(x) in ' + r'$\sigma$')
ax3d.xaxis.label.set_fontsize(15)
ax3d.xaxis.labelpad = 20
ax3d.set_zlabel(r'$\pi_{\theta}$' + "(x*)") # prob of greedy selection which has been min max scaled
ax3d.zaxis.label.set_fontsize(15)
ax3d.zaxis.label.set_rotation(180)
ax3d.zaxis.labelpad = 20

#flip z axis
# setting color as a function of z
ax3d.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='viridis')

plt.show()