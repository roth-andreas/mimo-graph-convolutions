# Plot Single-Channel Convolution
import matplotlib.pyplot as plt
import numpy as np

n = 5
np.random.seed(2)
plt.rcParams.update({'font.size': 14})
# Plot filter

x = np.arange(n)
filter = np.random.randn(n) * 0.8

fig, ax = plt.subplots(figsize=(3,2))
markers, stems, base = ax.stem(x, filter)
stems.set_linewidth(3)
markers.set_markersize(7)
ax.set_xticks(np.arange(n))
ax.set_ylim([-2,2])
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Coefficient')
fig.show()
fig.savefig('./figures_old/single_channel_filter.svg', bbox_inches='tight')


# Plot signal

signal = np.random.randn(n) * 0.8

fig, ax = plt.subplots(figsize=(3,2))
markers, stems, base = ax.stem(x, signal)
stems.set_linewidth(3)
markers.set_markersize(7)
ax.set_xticks(np.arange(n))
ax.set_ylim([-2,2])
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Coefficient')
fig.show()
fig.savefig('./figures_old/single_channel_signal.svg', bbox_inches='tight')

# Plot signal

z = filter * signal

fig, ax = plt.subplots(figsize=(3,2))
markers, stems, base = ax.stem(x, z)
stems.set_linewidth(3)
markers.set_markersize(7)
ax.set_xticks(np.arange(n))
ax.set_ylim([-2,2])
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Coefficient')
fig.show()
fig.savefig('./figures_old/single_channel_convolution.svg', bbox_inches='tight')