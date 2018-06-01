library (acepack)  #dCov
library (energy)  #dcov

# first filter by 5000 cells then do 21k cells
# for fast filter renyi > 0.85, dCov > 0.1

# Parameters
fname = '10xHumanPbmc.g949.rpm.log.csv'
mode = 'dCov'  # 'renyi', 'dCov'
subsampling = F  # T/F
subsample_size = 2000 # 5000
fast_sample_size = 2000
fast_min_nl_corr = 0.1

# Thresholds
min_samples = 200 # 10

min_pearsonr = -0.1 #disabled
max_pearsonr = 0.4 # for faster processing

min_nl_corr_xy = 0.1 # 0.1 if zeros not excluded #flag1
nonlinear_ratio = 1.1 # 1 if zeros not excluded  #flag2

# folders
getwd()
dir.create('./flag1')
dir.create('./flag2')

# read csv
df = read.csv(fname, header = T, row.names = 1)
dim(df)
df[0:2, 0:4]

# subsampling
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
print(paste('in program data size:', dim(df)))

# small fast matrix
df_fast = df[, sample(1:n_samples, fast_sample_size)]

# main
for (i in 1:n_gene){
      genei =  row.names(df[i,])
      print(paste('i:', i, genei))
      
      x = t(df[i, ])
      x[x == 0] <- NA
      x_sd = sd(x, na.rm=T)
      if (is.na(x_sd) || x_sd == 0){
            print('next: x_sd issue')
            next
      }
      
      for (j in 1:n_gene){
            genej =  row.names(df[j,])
            print(paste('i:', i, 'j:', j, genei, genej))
            # Sys.sleep(1)
            y = t(df[j,])
            y[y == 0] <- NA
            y_sd = sd(y, na.rm=T)
            if (i == j){
                  print('i equals j')
                  next
            }
            if (is.na(y_sd)||y_sd == 0){
                  print('next: y_sd issue')
                  next
            }
            
            # Remove NA if one in genei/genej is NA
            xy = df[c(i, j),] 
            xy[xy == 0] = NA
            xy = t(xy)
            xy = na.omit(xy)
            xy = t(xy)
            if (dim(xy)[2] < min_samples){
                  print(paste('next: xy_too_short issue, dim(xy)', dim(xy)))
                  next
            }
            x = xy[1,]
            y = xy[2,]

            # remove NA from df_fast
            xy_fast = df_fast[c(i, j),] 
            xy_fast[xy_fast == 0] = NA
            xy_fast = t(xy_fast)
            xy_fast = na.omit(xy_fast)
            xy_fast = t(xy_fast)
            if (dim(xy_fast)[2] < 20){
                  print(paste('next: xy_fast_too_short issue, dim(xy)', dim(xy), dim(xy_fast)))
                  next
            }
            x_fast = xy_fast[1,]
            y_fast = xy_fast[2,]
            
            # Cal corrs
            pearson_xy <- round(cor(x, y), 3)
            if (abs(pearson_xy) > max_pearsonr || abs(pearson_xy) < min_pearsonr){
                  print('pearsonr pass')
                  next
            }
            
            if (mode == 'dCov'){
                  fast_nl_corr = round(dcov(x_fast, y_fast), 4)
                  print(paste('i:', i, 'j:', j, genei, genej, 'fast_dcov', fast_nl_corr))
                  if (abs(fast_nl_corr) < fast_min_nl_corr){
                        print(paste('fast nl_corr too small'))
                        next
                  }
                  dCov_xy <- round(dcov(x, y), 3) #slow
                  nl_corr_xy = dCov_xy
            } else if (mode == 'renyi'){
                  ace_xy_fast <- ace(x_fast, y_fast)
                  fast_nl_corr = round(ace_xy_fast$rsq, 4)
                  print(paste('i:', i, 'j:', j, genei, genej, 'fast_renyi', fast_nl_corr))
                  if (abs(fast_nl_corr) < fast_min_nl_corr){
                        print(paste('fast nl_corr too small'))
                        next
                  }
                  ace_xy <- ace(x,y)
                  renyi_xy <- round(ace_xy$rsq, 3)
                  nl_corr_xy = renyi_xy
            } else {
                  stop('mode err')
            }

            if(any(is.na(c(nl_corr_xy, pearson_xy)))){
                  print('corr_any_na issue')
                  print(paste(nl_corr_xy, pearson_xy))
                  print(paste('sample size: ', dim(xy)))
                  next
            }

            cat(paste('result,', i, j, genei, genej, fast_nl_corr, nl_corr_xy, pearson_xy, '\n', sep = ','),
                  file="result.csv",append=TRUE)

            large_nl_to_l_ratio = abs(nl_corr_xy) > abs(pearson_xy)*nonlinear_ratio
            large_nl_corr = abs(nl_corr_xy) > min_nl_corr_xy
            flag1 = large_nl_corr 
            flag2 = flag1 && large_nl_to_l_ratio
            
            if(flag1){
                  print(paste('flag1: i:', i, 'j:', j, genei, genej))
                  cat(paste(i, j, genei, genej, nl_corr_xy, pearson_xy, 'end\n', sep = ','),
                      file="flag1.csv",append=TRUE)
                  
                  png(paste('./flag1/', genei,'_', genej,'.png', sep = ''))
                  plot (x, y, pch=16, cex=1, col="blue", 
                        xlab=paste(mode , nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                   genei, genej ))
                  dev.off()
                  
                  x2 = t(df[i, ])
                  y2 = t(df[j, ])
                  png(paste('./flag1/', genei,'_', genej,'.full.png', sep = ''))
                  plot (x2, y2, pch=16, cex=1, col="blue", 
                        xlab=paste(mode , nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                   genei, genej ))
                  dev.off()
                  
            }
            
            if (flag2){
                  print(paste('flag2: i:', i, 'j:', j, genei, genej))
                  cat(paste(i, j, genei, genej, nl_corr_xy, pearson_xy, 'end\n', sep = ','),
                      file="flag2.csv",append=TRUE)

                  png(paste('./flag2/', genei,'_', genej,'.png', sep = ''))
                  plot (x, y, pch=16, cex=1, col="blue", 
                        xlab=paste(mode, nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                   genei, genej ))
                  dev.off()
                  
                  png(paste('./flag2/', genei,'_', genej,'.full.png', sep = ''))
                  plot (x2, y2, pch=16, cex=1, col="blue", 
                        xlab=paste(mode, nl_corr_xy, ', pearsonr', pearson_xy, '\n',
                                   genei, genej ))
                  dev.off()
            }
      }
}


