# Hyper structure #
stage = 'step1'  # step1/step2
L = 7  # only a reporter, changing it can't alter the model structure
l = L//2
n_hidden_1 = 800
n_hidden_2 = 400  # update for different depth
n_hidden_3 = 200
# n_hidden_4 = 100 # add more after changing model structure

# Training parameters #
pIn = 0.8
pHidden = 0.5
learning_rate = 3e-4  # step1: 3e-4 for 3-7L, 3e-5 for 9L
sd = 1e-4  # step1: 3-7L:1e-3, 9L:1e-4
batch_size = 256
max_training_epochs = int(3e3)  # step1, EMT data: 3L:100, 5L/7L:1000, 9L:3000
display_step = 50  # interval on learning curve
snapshot_step = 1000  # interval of saving session, imputation
[a, b, c] = [0.7, 0.15, 0.15]  # splitting proportion: train/valid/test

# file input #
# EMT.MAGIC
# file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.A.log.hd5"  # input
# name1 = '(EMT_MAGIC_A)'
# file_orientation = 'gene_row'  # cell_row/gene_row

# GTEx
file1 = "../../../../data/gtex/tpm/gtex_v7.log.hd5"  # input
name1 = '(gtex_tpm_log_all)'  # uses 20GB of RAM
file_orientation = 'gene_row'  # cell_row/gene_row

# For Test Run #
seed_tf = 3
test_flag = 1  # {0, 1}, in test mode only 10000 gene, 1000 cells tested
if test_flag == 1:
    max_training_epochs = 10  # 3L:100, 5L:1000, 7L:1000, 9L:3000
    display_step = 1  # interval on learning curve
    snapshot_step = 5  # interval of saving session, imputation
    m = 1000
    n = 5500


# print parameters
print('Files:')
print('file1:', file1)
print('name1:', name1)
print('data_frame_orientation:', file_orientation)
print()

print('Parameters:')
print('stage:', stage)
print('test_mode:', test_flag)
print('{}L'.format(L))
print('{} Genes'.format(n))
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
