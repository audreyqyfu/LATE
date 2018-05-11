# Title     : Run Saver1.0
# Objective : Compare with Saver1.0
# Created by: rui
# Created on: 5/9/18

# devtools::install_github("mohuangx/SAVER")
library('SAVER')
getwd()

# read data (COUNT)
in_file = '~/data/cell_row/pbmc.g949_c10k.msk90.csv.gz'
out_file = 'pbmc_g949_c10k.msk.saver.csv.gz'
df = read.csv(in_file, row.names=1)
df[0:3, 0:3]

# single core
# saver4 <- saver(df)

# run in parallel
library(doParallel)
cl <- makeCluster(4, outfile = "")
registerDoParallel(cl)
saver5 <- saver(df)
gc(verbose=TRUE) #RAM usage
stopCluster(cl)


# save result
z <- gzfile(out_file)
write.csv(saver5$estimate, z)

