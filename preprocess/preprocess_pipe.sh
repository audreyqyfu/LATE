# Preprocess GTEx data
# Author: Boxiang Liu 
# Email: bliu2@stanford.edu

# Setup: 
mkdir -p ../processed_data/preprocess/


# Normalization:
python preprocess/normalization.logXplus1.py \
../data/GTEx/rna-seq/GTEx_Analysis_v7_RNA-seq_RNA-SeQCv1.1.8_gene_reads.gct.gz \
../processed_data/preprocess/GTEx_v7_RNA-SeQC