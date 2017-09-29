# clustering and heatmap
# hclust, and heat map
library(stats)
library(Matrix)
options(digits=3) #for print
#setwd("/Users/rui/AUDREY_LAB/imputation/tf/autoencoder/v2.ReLuMSE/v2.3.rand/v2.3.1.log/early_stop/heatmap")
rerun.flag = 1

#load data
#load("GSE72857.umitab.m0.RData")
file1 = 'encoder_w1.csv'
file2 = 'decoder_w2.csv'

library(data.table) #much faster for reading csv
m1 = as.matrix(fread(file1, header = F, sep = ','))
m2 = as.matrix(fread(file2, header = F, sep = ','))

# file1 #
#percent.nnzero
n.genes.1=dim(m1)[1]
n.cells.1=dim(m1)[2]
n.nnzero.1=nnzero(m1)
percent.nnzero.1 = n.nnzero.1/length(m1)

sink(paste(file1,"data_desc.txt", sep = "."))
cat("number of input genes:  ",n.genes.1)
cat("\nnumber of input cells:  ",n.cells.1)
cat("\npercent.nnzero:  ", percent.nnzero.1)
sink()

#hist reads/cell(colSums)  reads/gene(rowSums)
reads.cell = colSums(m1)
png(filename = paste(file1, "hist.cell.count.png", sep="."))
hist(log10(reads.cell+1),breaks = 100) #on log scale, reads/cell looks normal
dev.off()

reads.gene = rowSums(m1)
png(filename = paste(file1, "hist.gene.count.png", sep="."))
hist(log10(reads.gene+1),breaks = 100) #normal plus negative tail to zero
dev.off()

#Tree by cells
if(rerun.flag){
      load(paste(file1,"dist.cells.RData",sep = "."))
      load(paste(file1,"tree.cells.RData",sep = "."))
}else{
      dist.cells.1 = dist(t(m1)) #dist of cells
      save(dist.cells.1,file=paste(file1,"dist.cells.RData",sep = "."))
      tree.cells.1 = hclust(dist.cells.1,method="complete") 
      save(tree.cells.1,file=paste(file1,"tree.cells.RData",sep = "."))
}

png(filename= paste(file1, "tree.cell.png", sep="."))
plot (tree.cells.1, labels = F, main = "Cluster Dendrogram by Cells", 
      sub = paste("[genes,cells]=",n.genes.1,n.cells.1 ))
dev.off()

#Tree by genes
if(rerun.flag){
      load(paste(file1,"dist.genes.RData", sep="."))
      load(paste(file1,"trees.genes.RData",sep='.'))
}else{
      dist.genes.1 = dist(m1) #dist of genes (by rows)
      save(dist.genes.1,file= paste(file1,"dist.genes.RData", sep=".") )
      tree.genes.1 = hclust(dist.genes.1,method="complete") 
      save(tree.genes.1,file=paste(file1,"trees.genes.RData",sep='.'))
}

png(filename= paste(file1, "tree.genes.png", sep="."))
plot (tree.genes.1, labels = F, main = "Cluster Dendrogram by Genes",
      sub = paste("[genes,cells]=",n.genes.1,n.cells.1 ))
dev.off()

#Heatmap2 # (put hclust.tree in it, change color scheme)
library(gplots)
png(filename= paste(file1, "heatmap2.png", sep="."), 
    width=n.genes.1+300, height=n.genes.1+300)
heatmap.2(m1,Rowv = as.dendrogram(tree.genes.1), Colv = as.dendrogram(tree.cells.1),
          trace = 'none', 
          col=colorRampPalette(c("green", 'white', "red"))(n = 100)
)
dev.off()

# file2 #
#percent.nnzero
n.genes.2 = dim(m2)[1] 
n.cells.2 = dim(m2)[2] 
n.nnzero.2 = nnzero(m2)
percent.nnzero.2 = n.nnzero.2/length(m2)

sink(paste(file2,"data_desc.txt", sep = "."))
cat("number of input genes:  ",n.genes.2)
cat("\nnumber of input cells:  ",n.cells.2)
cat("\npercent.nnzero:  ", percent.nnzero.2)
sink()

#hist reads/cell(colSums)  reads/gene(rowSums)
reads.cell.2 = colSums(m2)

png(filename = paste(file2, "hist.cell.count.png", sep="."))
hist(log10(reads.cell.2+1),breaks = 100) #most less than 10e4 reads/cell
dev.off()

reads.gene.2 = rowSums(m2)
png(filename = paste(file2, "hist.gene.count.png", sep="."))
hist(log10(reads.gene.2+1),breaks = 100) #some genes with little expression in most cells
dev.off()

#Tree by cells
if(rerun.flag){
      load(paste(file2,"dist.cells.RData",sep = "."))
      load(paste(file2,"tree.cells.RData",sep = "."))
}else{
      dist.cells.2 = dist(t(m2)) #dist of cells
      save(dist.cells.2,file=paste(file2,"dist.cells.RData",sep = "."))
      tree.cells.2 = hclust(dist.cells.2,method="complete") 
      save(tree.cells.2,file=paste(file2,"tree.cells.RData",sep = "."))
}

png(filename= paste(file2, "tree.cell.png", sep="."))
plot (tree.cells.2, labels = F, main = "Cluster Dendrogram by Cells", 
      sub = paste("[genes,cells]=",n.genes.2,n.cells.2 ))
dev.off()

#Tree by genes
if(rerun.flag){
      load(paste(file2,"dist.genes.RData", sep="."))
      load(paste(file2,"trees.genes.RData",sep='.'))
}else{
      dist.genes.2 = dist(m2) #dist of genes (by rows)
      save(dist.genes.2,file= paste(file2,"dist.genes.RData", sep=".") )
      tree.genes.2 = hclust(dist.genes.2, method="complete") 
      save(tree.genes.2,file=paste(file2,"trees.genes.RData",sep='.'))
}

png(filename= paste(file2, "tree.genes.png", sep="."))
plot (tree.genes.2, labels = F, main = "Cluster Dendrogram by Genes",
      sub = paste("[genes,cells]=",n.genes.2,n.cells.2 ))
dev.off()

#Heatmap2 (put hclust.tree of file2 in it)
png(filename= paste(file2, "heatmap2.png", sep="."), 
    width=n.genes.2+300, height=n.genes.2+300)
heatmap.2(m2,Rowv = as.dendrogram(tree.genes.2), Colv = as.dendrogram(tree.cells.2),
        col=colorRampPalette(c("green", 'white', "red"))(n = 100), trace = 'none'
)
dev.off()

#Heatmap2 (put hclust.tree of file1 in it)
png(filename= paste(file2, "heatmap2.compare.png", sep="."), 
    width=n.genes.2+300, height=n.genes.2+300)
heatmap.2(m2,Rowv = as.dendrogram(tree.genes.1), Colv = as.dendrogram(tree.cells.1),
          col=colorRampPalette(c("green", 'white', "red"))(n = 100), trace = 'none'
)
dev.off()

