# source("https://bioconductor.org/biocLite.R")
# biocLite("splatter")
library(splatter)

# V2 One Group (Cell type)
# large libsize, set by hand
setwd("~/AUDREY_LAB/imputation/simulation/splatter/v1-1.Splat.OneGroupSet2/")

n_genes = 1000
n_cells = 20000
params <- newSplatParams(nGenes=n_genes, 
                         groupCells=c(n_cells),
                         lib.loc=15, 
                         dropout.present=FALSE,
                         seed=2
)

# simulation
sim <- splatSimulate(params, verbose = TRUE)

# Overal Plots
png(paste("PCA","png", sep="." ) )
plotPCA(sim)
dev.off()

png(paste("matrix_hist","png", sep="." ) )
hist(log10(counts(sim) +1), xlab='log(expression+1)')
dev.off()

# correlation among cells
corr = cor(counts(sim)[,1:2000] )
diag(corr)  = NA
png(paste("Corr_cells","png", sep="." ) )
hist(corr)
dev.off()


# Overall summary
sink(file="splat.OneGroup.params.txt")
params
cat("#Summary\ndim of matrix: ", dim(counts(sim)), "\n\n" )
paste("zero.percent=",
      format(sum(counts(sim)==0) / (nrow(counts(sim) )*ncol(counts(sim)) ), digits = 3)
)
cat("\n#Counts\n")
counts(sim)[1:10, 1:10]
cat("\n#fData\n")
head(fData(sim))
cat("\n#pData\n")
head(pData(sim))
sink()

# Save counts
saveRDS(counts(sim), "splat.OneGroup.RData")

# Gene-Gene Relationship
set.seed(0)
gene_list = as.list(sample(1:n_genes, 10) )
gene_pairs = matrix(gene_list, ncol=2)

for (i in 1:dim(gene_pairs)[1]){
      cat(i)
      # get data
      name_x = paste("Gene", gene_pairs[i,1], sep="")
      name_y = paste("Gene", gene_pairs[i,2], sep="")
      x = log10(counts(sim)[name_x,]+1)
      y = log10(counts(sim)[name_y,]+1)
      # boxplot
      png(paste(name_x, name_y, "box", "png", sep="." ) )
      boxplot(x, y, main=paste('Boxplot of', name_x, name_y) )
      dev.off()
      # hist x
      png(paste(name_x,"hist","png", sep="." ) )
      hist(x, main=name_x, xlab='log(expression+1)' )
      dev.off()
      # hist y
      png(paste(name_y,"hist","png", sep="."))
      hist(y, main=name_y, xlab='log(expression+1)')
      dev.off()
      # biaxial plot
      png(paste(name_x, name_y,"png", sep="." ) )
      plot(x,y, 
           xlab=name_x, ylab=name_y,
           main="Gene-gene biaxial plot\nlog(x+1) transformed",
           sub=paste("cor=", format(cor(x,y), digit=3) )
           )
      dev.off()
}

# save csv
write.csv(x = counts(sim), file = "splat.OneGroup.csv")
