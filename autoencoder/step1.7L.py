import pandas as pd

# Variables:
DATA_FN='../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5'

# Functions:
def read_hdf5(fn):
	df = pd.read_hdf(fn).transpose()
	return df

df=read_hdf5(DATA_FN)
df.iloc[1:5,1:5]