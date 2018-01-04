import pandas as pd
import scimpute

df = scimpute.read_hd5('v1-1-5-3.F3.msk.hd5')
df.to_csv('v1-1-5-3.F3.msk.csv.gz', float_format='%.6f', compression='gzip')

df = scimpute.read_hd5('v1-1-5-3.F3.hd5')
df.to_csv('v1-1-5-3.F3.csv.gz', float_format='%.6f', compression='gzip')


df = scimpute.read_hd5('v1-1-5-3.E3.hd5')
df.to_csv('v1-1-5-3.E3.csv.gz', float_format='%.6f', compression='gzip')

# df = pd.read_hdf('v1-1-5-3.F3.msk.hd5')
# df.to_csv('v1-1-5-3.F3.msk.csv.gz', float_format='%.6f', compression='gzip')

# df = pd.read_hdf('v1-1-5-3.F3.hd5')
# df.to_csv('v1-1-5-3.F3.csv.gz', float_format='%.6f', compression='gzip')
