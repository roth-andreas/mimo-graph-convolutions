import numpy as np
import matplotlib.pyplot as plt

n = 5
d = 3
c = 4
np.random.seed(2)
plt.rcParams.update({'font.size': 18})

x = np.array([(np.arange(n)-i) % n for i in range(d)]).flatten()
y = np.repeat(np.arange(d), n)
z = np.random.randn(d*n)
signal = [np.random.uniform(size=d) for i in range(n)]

fig, ax = plt.subplots(figsize=(5,5),subplot_kw=dict(projection='3d'))
#ax.stem(x, y, z)
for i in range(n):
  markers, stems, base = ax.stem(np.array([i]*d), np.arange(d), signal[i]*2-1)
  stems.set_linewidth(3)
  markers.set_markersize(7)
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(d))
ax.set_zticks([-0.5,0,0.5])
#ax.set_xticklabels(np.arange(n))
ax.set_ylim([0,d-1])
ax.set_xlim([0,n-1])
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Input channel index')
ax.set_zlabel('Coefficient', labelpad=10)
fig.subplots_adjust(left=-0.02, right=0.88, bottom=0.05, top=1.0)
fig.show()
fig.savefig('./figures_old/multi_channel_signal.svg')#, bbox_inches='tight')



from matplotlib import cm
plt.rcParams.update({'font.size': 16})
x = np.array([(np.arange(n)-i) % n for i in range(d)]).flatten()
y = np.repeat(np.arange(d), n)
z = np.random.uniform(size=d*n)

fig, ax = plt.subplots(figsize=(5,5),subplot_kw=dict(projection='3d'))

point = np.array([1, 2, 3])
normal = np.array([1, 1, 2])

# create x,y
zz, yy = np.meshgrid(range(c+1), range(d+1))

yy = yy -0.5
zz = zz -0.5

filter = [np.random.uniform(size=(d+1,c+1)) for i in range(n)]

for i in range(n):
  # calculate corresponding z
  xx = np.ones_like(zz) * i

  Gx, Gy = np.gradient(zz * yy)  # gradients with respect to x and y
  G = (Gx ** 2 + Gy ** 2) ** .5  # gradient magnitude
  N = G / G.max()  # normalize 0..1


  ax.plot_surface(xx, yy, zz, facecolors=cm.viridis(filter[i]))


ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(d+1))
ax.set_zticks(np.arange(c+1))
#ax.set_xticklabels(np.arange(n))
ax.set_ylim([-0.5,d-0.5])
ax.set_zlim([-0.5,c-0.5])
ax.set_xlim([0,n-1])
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Input channel index')
ax.set_zlabel('Output channel index', labelpad=5)
fig.subplots_adjust(left=-0.04, right=0.88, bottom=0.05, top=1.0)
fig.show()
fig.savefig('./figures_old/multi_channel_filter.svg')#, bbox_inches='tight')


plt.rcParams.update({'font.size': 18})
x = np.array([(np.arange(n)-i) % n for i in range(d)]).flatten()
y = np.repeat(np.arange(d), n)
z = np.random.randn(d*n)

fig, ax = plt.subplots(figsize=(5,5),subplot_kw=dict(projection='3d'))
#ax.stem(x, y, z)
for i in range(n):
  markers, stems, base = ax.stem(np.array([i]*c), np.arange(c), np.random.uniform(size=c)*2-1)
  stems.set_linewidth(3)
  markers.set_markersize(7)
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(c))
#ax.set_xticklabels(np.arange(n))
ax.set_ylim([0,c-1])
ax.set_xlim([0,n-1])
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Output channel index')
ax.set_zlabel('Coefficient', labelpad=10)
fig.subplots_adjust(left=-0.02, right=0.88, bottom=0.05, top=1.0)
fig.show()
fig.savefig('./figures_old/multi_channel_convoluted.svg')#, bbox_inches='tight')