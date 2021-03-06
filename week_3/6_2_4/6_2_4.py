import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


w = np.linspace(-10,10,100)
v = np.array([2,1])

v = v[:,None] * w

# plotting v_w
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(v[0,:], v[1,:], w)
# plt.xlabel('v(s1)')
# plt.ylabel('v(s2)')
# ax.set_zlabel('w')
# ax.set_ylim(-20,20)
# ax.set_xlim(-20,20)
# plt.savefig('./v_w.pdf')
# plt.close()

# B^\pi v_w is simply v_w with the roles of s1 and s2 exchanged, since we have zero discounting and zero reward 
# plotting B^pi v_w
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(v[1,:], v[0,:], w)
# plt.xlabel('v(s1)')
# plt.ylabel('v(s2)')
# ax.set_zlabel('w')
# ax.set_ylim(-20,20)
# ax.set_xlim(-20,20)
# plt.savefig('./b_v.pdf')
# plt.close()

# computing \Pi B^\pi v_w
pi_b_v = 4/5*v
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(pi_b_v[0,:], pi_b_v[1,:], w)
# plt.xlabel('v(s1)')
# plt.ylabel('v(s2)')
# ax.set_ylim(-20,20)
# ax.set_xlim(-20,20)
# ax.set_zlabel('w')
# plt.savefig('./pi_b_v.pdf')
# plt.close()


# plotting all in one
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(v[0,:], v[1,:], w, color='r', label=r'$v_w$')
plt.plot(pi_b_v[0,:], pi_b_v[1,:], w, color='g', label=r'$\Pi B^{\pi} v_w$')
plt.plot(v[1,:], v[0,:], w, color='b', label=r'$B^{\pi} v_w$')
plt.legend()
plt.xlabel('v(s1)')
plt.ylabel('v(s2)')
ax.set_ylim(-2.5,2.5)
ax.set_xlim(-2.5,2.5)
ax.set_zlim(-2.5, 2.5)
ax.set_zlabel('w')

X, Y = np.meshgrid(range(-3, 4), range(-3, 4))
Z = np.ones((7, 7))
ax.plot_surface(X, Y, Z, color="purple", alpha=0.3)
ax.scatter([2, 1, 4/5*2], [1, 2, 4/5], [1, 1, 1], color="purple")
#ax.scatter(4/5*2, 4/5*1, 4/5, color="black")
ax.plot([2, 1], [1, 2], [1, 1], color="purple", linestyle="dashed")
ax.plot([1, 4/5*2], [2, 4/5], [1, 1], color="purple", linestyle="dashed")
#ax.plot([4/5*2, 4/5*2], [4/5, 4/5], [1, 4/5], color="black", linestyle="dashed")
plt.savefig('./all.pdf')
plt.show()
