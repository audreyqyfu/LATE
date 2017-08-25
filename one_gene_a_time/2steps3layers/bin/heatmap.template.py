import matplotlib
import matplotlib.pyplot as plt
import numpy as np

m = np.random.rand(10,10)
x = np.random.rand(1,m.shape[1])
y = np.random.rand(m.shape[0],1)

norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
grid = dict(height_ratios=[1, m.shape[0]], width_ratios=[1,m.shape[0], 0.5 ])
fig, axes = plt.subplots(ncols=3, nrows=2, gridspec_kw = grid)

axes[1,1].imshow(m, aspect="auto", cmap="viridis", norm=norm)
axes[0,1].imshow(x, aspect="auto", cmap="viridis", norm=norm)
axes[1,0].imshow(y, aspect="auto", cmap="viridis", norm=norm)

axes[0,0].axis("off")
axes[0,2].axis("off")

axes[1,1].set_xlabel('Number 1')
axes[1,1].set_ylabel('Number 2')
for ax in [axes[1,1], axes[0,1], axes[1,0]]:
    ax.set_xticks([]); ax.set_yticks([])

sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

fig.colorbar(sm, cax=axes[1,2])

plt.show()