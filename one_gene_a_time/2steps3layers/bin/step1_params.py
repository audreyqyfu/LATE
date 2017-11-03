# Hyper structure #
L = 7  # only a reporter, changing it can't alter the model structure
l = L//2
n_hidden_1 = 800
n_hidden_2 = 400  # update for different depth
n_hidden_3 = 200
n_hidden_4 = -1  # -1 means no such layer

# Training parameters #
pIn = 0.8
pHidden = 0.5
learning_rate = 0.0003  # 0.0003 for 3-7L, 0.00003 for 9L, update for different depth
sd = 0.0001  # 3-7L:1e-3, 9L:1e-4, update for different depth
batch_size = 256
training_epochs = 1000  #3L:100, 5L:1000, 7L:1000, 9L:3000
display_step = 20  # interval on learning curve
snapshot_step = 500  # interval of saving session, imputation
[a, b, c] = [0.7, 0.15, 0.15]  # splitting proportion: train/valid/test

# file input #
# EMT.MAGIC
file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # input
file2 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # ground truth (same as input in step1)
name1 = '(EMT_MAGIC_A)'
name2 = '(EMT_MAGIC_A)'

# GTEx
# file1 = "../../../../data/gtex/gtex_v7.norm.log.hd5"  # input
# file2 = "../../../../data/gtex/gtex_v7.norm.log.hd5"  # ground truth (same as input in step1)
# name1 = '(gtex_gene)'  # uses 20GB of RAM
# name2 = '(gtex_gene)'

# For development usage #
seed_tf = 3
test_flag = 0  # [0, 1], in test mode only 10000 gene, 1000 cells tested
if test_flag == 1:
    training_epochs = 10  # 3L:100, 5L:1000, 7L:1000, 9L:3000
    display_step = 1  # interval on learning curve
    snapshot_step = 5  # interval of saving session, imputation