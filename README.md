# scImpute
# transAutoencoder: autoencoder structure, pre-trained on A, re-train on B.msk
# workflow
- step1: scImpute/one_gene_a_time/2steps3layers/bin/step1.n_to_n.new.7L.py
-- command: python step1.n_to_n.new.7L.py

- development environment:
-- python version: 3.5.2 (default, Dec 13 2016, 14:11:32)
-- tf.__version__ 1.2.1
