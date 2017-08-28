# modified from : https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib


# matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import gaussian_kde

def density_plot(x, y, title, fname):
	# create plots directory
	if not os.path.exists("plots"):
		os.makedirs("plots")
	fname = "./plots/"+fname
	# Calculate the point density
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	# sort: dense on top (plotted last)
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	#plt
	#fig = plt.figure(figsize=(9,9))
	fig, ax = plt.subplots()
	cax = ax.scatter(x, y, c=z, s=50, edgecolor='')
	plt.title(title)
	plt.colorbar(cax)
	plt.savefig(fname+".png", bbox_inches='tight')
	plt.close(fig)


# input data
in_name = 'splat.OneGroup.norm.hd5'
df = pd.read_hdf(in_name)
print (str(in_name) + str(df.shape))

# gene-gene biaxial plot (density colored)
print("Plotting:");

m = 3
np.random.seed(0)
num_arr = np.random.choice(1000, [m, 2], replace=False)

for i in np.arange(m):
	name1= str('Gene'+str(num_arr[i][0]))
	name2= str('Gene'+str(num_arr[i][1]))

	x = df.loc[name1]
	y = df.loc[name2]
	corr_df = df.loc[[name1,name2]].transpose().corr()
	corr = round(corr_df.ix[1,0],3)
	title = name1+name2+' corr='+str(corr)
	fname = name1+name2+".biaxial"
	density_plot(x, y, title, fname);

