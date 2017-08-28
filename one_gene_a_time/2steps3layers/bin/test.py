import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scimpute

weight = np.random.normal(0,10, [100, 50])
bias = np.random.normal(0,20, [1, 50])

vmax_w, vmin_w = scimpute.max_min_element_in_arrs([weight])
vmax_b, vmin_b = scimpute.max_min_element_in_arrs([bias])

norm_w = matplotlib.colors.Normalize(vmin=vmin_w, vmax=vmax_w)
norm_b = matplotlib.colors.Normalize(vmin=vmin_b, vmax=vmax_b)

grid = dict(height_ratios=[weight.shape[0], weight.shape[0]/40, weight.shape[0]/40],
            width_ratios=[weight.shape[1], weight.shape[1]/40 ])
fig, axes = plt.subplots(ncols=2, nrows=3, gridspec_kw=grid)

axes[0, 0].imshow(weight, aspect="auto", cmap="PiYG", norm=norm_w)
axes[1, 0].imshow(bias, aspect="auto", cmap="PiYG", norm=norm_b)

for ax in [axes[0,0]]:
    ax.set_xticks([])

for ax in [axes[1,0]]:
    ax.set_yticks([])

for ax in [axes[1,1], axes[2,1]]:
    ax.axis("off")

# axes[1, 0].set_xlabel('node out')
# axes[1, 0].set_ylabel('node in')


sm_w = matplotlib.cm.ScalarMappable(cmap="PiYG", norm=norm_w)
sm_w.set_array([])
sm_b = matplotlib.cm.ScalarMappable(cmap="PiYG", norm=norm_b)
sm_b.set_array([])

fig.colorbar(sm_w, cax=axes[0,1])
fig.colorbar(sm_b, cax=axes[2,0], orientation="horizontal")

plt.show()