import os
home = os.environ['HOME']

# Hyper structure #
stage = 'step2'  # step1/step2
L = 5  # only a reporter, changing it can't alter the model structure
l = L//2
n_hidden_1 = 400
n_hidden_2 = 200  # update for different depth
# n_hidden_3 = 200
# n_hidden_4 = 100 # add more after changing model structure

run_flag = 'rand_init'  # rand_init/load_saved

# Training parameters #
pIn = 0.8
pHidden = 0.5
if run_flag == 'rand_init':
    learning_rate = 3e-4  # step1: 3e-4 for 3-7L, 3e-5 for 9L
elif run_flag == 'load_saved':
    learning_rate = 3e-5  # step2: 3e-5 for 3-7L, 3e-6 for 9L
reg_coef = 1e-2  # reg3, can set to 0.0
sd = 1e-3  # 3-7L:1e-3, 9L:1e-4, update for different depth
batch_size = 256
max_training_epochs = int(1e3)
display_step = 50  # interval on learning curve
snapshot_step = int(5e2)  # interval of saving session, imputation
[a, b, c] = [0.7, 0.15, 0.15]  # splitting proportion: train/valid/test
patience = 5  # early stop patience epochs, just print warning, early stop not implemented yet

# PBMC
file1 = home+'/imputation/data/10x_human_pbmc_68k/filtering/rpm/msk/\
10xHumanPbmc.g5561.rpmLog.msk98.hd5'
name1 = 'PBMC.G5561.RPM.LOG.MSK98'
file1_orientation = 'gene_row'  # cell_row/gene_row
file2 = home+'/imputation/data/10x_human_pbmc_68k/filtering/rpm/\
10xHumanPbmc.g5561.rpmLog.hd5'
name2 = 'PBMC.G5561.RPM.LOG'
file2_orientation = 'gene_row'  # cell_row/gene_row

# For development usage #
seed_tf = 3
test_flag = 1  # [0, 1], in test mode only 10000 gene, 1000 cells tested
if test_flag == 1:
    max_training_epochs = 20 # 3L:100, 5L:1000, 7L:1000, 9L:3000
    display_step = 2  # interval on learning curve
    snapshot_step = 10  # interval of saving session, imputation
    m = 1000
    n = 300

# Gene list (data frame index)
pair_list = [
            [2, 3],
            [205, 206]
            ]

gene_list = [
    2, 3, 205, 206]


# print parameters
print('Files:')
print('file1:', file1)
print('name1:', name1)
print('file2:', file2)
print('name2:', name2)
print('data_frame1_orientation:', file1_orientation)
print('data_frame2_orientation:', file1_orientation)

print()

print('Parameters:')
print('stage:', stage)
print('test_mode:', test_flag)
print('{}L'.format(L))
# print('{} Genes'.format(n))
for l_tmp in range(1, l+1):
  print("n_hidden{}: {}".format(l_tmp, eval('n_hidden_'+str(l_tmp))))

print('learning_rate:', learning_rate)
print('batch_size:', batch_size)
print('pIn:', pIn)
print('pHidden:', pHidden)
print('max_training_epochs:', max_training_epochs)
print('display_step', display_step)
print('snapshot_step', snapshot_step)
print()

