library(acepack)
library(energy)
library(scales)  # for alpha in plotting

# Parameters
fname = '/Volumes/radio/audrey2/imputation/data/10x_human_pbmc_68k/10x_human_pbmc_68k.csv'
mode = 'renyi'  # 'renyi', 'dCov'
subsampling = T  # T/F
subsample_size = 5000 # renyi: 5000 dCov: 1000
matrix_direction = 'gene_row'  # cell_row

# Thresholds
min_samples = 200 # 200

min_pearsonr = -0.1 #disabled
max_pearsonr = 0.4 #0.4

min_nl_corr_xy = 0.6 #flag1: renyi: 0.4-0.6
min_nl_ratio = 1.5 #flag2: renyi: 1.1-2

max_figs = 500

# folders
getwd()
dir.create('./flag1')
dir.create('./flag2')

# read csv
df = read.csv(fname, header = T, row.names = 1)
if (matrix_direction == 'cell_row'){
      df = t(df)
}  # gene_row in this script

cat('genes, cells', dim(df), '\n')
df[0:2, 0:2]

# subsampling
cat('taking sub-sample from cells, sample_size=', subsample_size, '\n')
n_gene = nrow(df)
n_samples = ncol(df)
if (subsampling){
      n_sub_samples = subsample_size  # take sub-samples for speed
      set.seed(1)
      df <- df[, sample(1:n_samples, n_sub_samples)]
      dim(df)
      df[0:2, 0:2]
      n_gene = nrow(df)
      n_samples = ncol(df)
}
cat('genes, cells', dim(df), '\n')
df[0:2, 0:2]

# main
count1 = 0
count2 = 0
# header of csv's
cat(paste('i', 'j', 'genei', 'genej', 'nl_corr_xy', 'pearson_xy', 'nl_ratio','\n', sep = ','),
    file="results.csv", append=TRUE)
cat(paste('i', 'j', 'genei', 'genej', 'nl_corr_xy', 'pearson_xy',
          'nl_ratio', '\n', sep = ','),
    file="flag1.large_nl_corr.csv",append=TRUE)
cat(paste('i', 'j', 'genei', 'genej', 'nl_corr_xy', 'pearson_xy',
          'nl_ratio', '\n', sep = ','),
    file="flag2.large_ratio_too.csv",append=TRUE)

# main loop
for (i in 1:n_gene){
      genei =  row.names(df)[i]
      cat('i:', i, genei, '\n')
      
      x = t(df[i, ])
      x[x == 0] <- NA
      x_sd = sd(x, na.rm=T)
      
      if (is.na(x_sd) || x_sd == 0){
            cat('next: genei sd is zero\n')
            next
      }
      
      for (j in 1:n_gene){
            if (i >= j){
                  print('i >= j')
                  next
            }
            
            genej =  row.names(df)[j]
            cat('i:', i, 'j:', j, genei, genej, '\n')
            
            y = t(df[j,])
            y[y == 0] <- NA
            y_sd = sd(y, na.rm=T)
            
            if (is.na(y_sd)||y_sd == 0){
                  cat('next: y_sd issue\n')
                  next
            }
            
            # Remove NA if one in genei/genej is NA
            xy = df[c(i, j),]
            xy[xy == 0] = NA
            xy = t(xy)
            xy = na.omit(xy)
            xy = t(xy)
            if (dim(xy)[2] < min_samples){
                  cat('next: xy_too_short issue, dim(xy)', dim(xy), '\n')
                  next
            }
            x = xy[1,]
            y = xy[2,]
            
            # Cal corrs
            pearson_xy <- round(cor(x, y), 3)
            if (is.na(pearson_xy)){
                  cat('pearson_xy is NA\n')
                  next
            }
            
            if (abs(pearson_xy) > max_pearsonr || abs(pearson_xy) < min_pearsonr){
                  cat('pearsonr too large or too small: ', pearson_xy)
                  next
            }
            
            if (mode == 'dCov'){
                  dCov_xy <- round(dcov(x, y), 3) #slow
                  nl_corr_xy = dCov_xy
            } else if (mode == 'renyi'){
                  ace_xy <- ace(x,y)
                  renyi_xy <- round(ace_xy$rsq, 3)
                  nl_corr_xy = renyi_xy
            } else {
                  stop('mode err')
            }
            
            if(any(is.na(nl_corr_xy))){
                  cat('nl_corr is NA\n')
                  next
            }
            
            # if there were no issues
            nl_ratio = abs(nl_corr_xy / pearson_xy)
            cat(paste(i, j, genei, genej, nl_corr_xy, pearson_xy, nl_ratio,'\n', sep = ','),
                file="results.csv", append=TRUE)
            
            # boolean
            large_nl_to_l_ratio = (nl_ratio > min_nl_ratio)
            large_nl_corr = (abs(nl_corr_xy) >= min_nl_corr_xy)
            
            flag1 = large_nl_corr
            flag2 = large_nl_corr && large_nl_to_l_ratio
            
            # plotting
            if(flag1){
                  count1 = count1 + 1
                  
                  print(paste('flag1: i:', i, 'j:', j, genei, genej))
                  
                  cat(paste(i, j, genei, genej, nl_corr_xy, pearson_xy,
                            nl_ratio, '\n', sep = ','),
                      file="flag1.large_nl_corr.csv",append=TRUE)
                  
                  if (count1 < max_figs){
                        png(paste('./flag1/', genei,'_', genej,'.png', sep = ''))
                        plot (x, y, pch=16, cex=1, col=alpha("blue", 0.2),
                              xlab=paste(mode , nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                         genei, genej ))
                        dev.off()
                        
                        # plot with all samples
                        x2 = t(df[i, ])
                        y2 = t(df[j, ])
                        png(paste('./flag1/', genei,'_', genej,'.full.png', sep = ''))
                        plot (x2, y2, pch=16, cex=1, col=alpha("blue", 0.2),
                              xlab=paste(mode, nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                         'in ', subsample_size, ' samples\n',
                                         genei, genej ))
                        dev.off()
                  }
                  
            }
            
            if (flag2){
                  count2 = count2 + 1
                  
                  print(paste('flag2: i:', i, 'j:', j, genei, genej))
                  cat(paste(i, j, genei, genej, nl_corr_xy, pearson_xy,
                            nl_ratio, '\n', sep = ','),
                      file="flag2.large_ratio_too.csv",append=TRUE)
                  
                  if (count2<max_figs){
                        png(paste('./flag2/', genei,'_', genej,'.png', sep = ''))
                        plot (x, y, pch=16, cex=1, col=alpha("blue", 0.2),
                              xlab=paste(mode, nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                         genei, genej ))
                        dev.off()
                        
                        # plot with all samples
                        png(paste('./flag2/', genei,'_', genej,'.full.png', sep = ''))
                        plot (x2, y2, pch=16, cex=1, col=alpha("blue", 0.2),
                              xlab=paste(mode, nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                         'in ', subsample_size, ' samples\n',
                                         genei, genej ))
                        dev.off()
                  }
            }
      }
}

