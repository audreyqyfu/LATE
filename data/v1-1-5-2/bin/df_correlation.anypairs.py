# compare correlation between two df

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import os
import time

file_1 = 'splat.OneGroup.norm.log.hd5'

# read data #
df1 = pd.read_hdf(file_1)

# calculate correlation between cells
pearsonrlog = []
range_size = 200
for i in range(range_size):
	for j in range(range_size):
		pearsonrlog.append(pearsonr(df1.values[:,i], df1.values[:,j])[0] )

pearsonrlog.sort()

# histogram of correlation
fig = plt.figure(figsize=(9,9))
plt.hist(pearsonrlog)
plt.xlabel('Pearson corr between cells')
plt.ylabel('Freq.')
plt.title("Correlation among any two cells in the original dataset \n(randomly selected 200 cells)")
plt.savefig("corr.hist.png", bbox_inches='tight')
plt.close(fig);






