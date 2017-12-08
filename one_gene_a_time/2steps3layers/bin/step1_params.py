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
sd = 1e-3  # step1: 3-7L:1e-3, 9L:1e-4
batch_size = 256
max_training_epochs = int(1e3)  # step1, EMT data: 3L:100, 5L/7L:1000, 9L:3000
display_step = 50  # interval on learning curve
snapshot_step = int(5e2)  # interval of saving session, imputation
[a, b, c] = [0.7, 0.15, 0.15]  # splitting proportion: train/valid/test
patience = 5  # early stop patience epochs, just print warning, early stop not implemented yet

# file input #
# EMT.MAGIC
file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.msk50.log.hd5"  # input
name1 = '(EMT_MAGIC_B.msk50)'
file_orientation = 'gene_row'  # cell_row/gene_row

# GTEx
# file1 = "../../../../data/gtex/rpm_4tissues/gtex_v7.rpm.log.muscle_heart_skin_adipose_no.hd5"  # input
# name1 = '(RPM_4TissuesNo)'  # uses 20GB of RAM for 5GB DF
file2 = file1
name2 = name1
# file_orientation = 'cell_row'  # cell_row/gene_row
file1_orientation = file_orientation  # cell_row/gene_row
file2_orientation = file_orientation

# Gene list
pair_list = [
            [4058, 7496],
            [8495, 12871],
            [2, 3],
            [205, 206]
            ]

gene_list = [
            4058, 7496, 8495, 12871,
            2, 3, 205, 206
             ]

# For Test Run #
seed_tf = 3
test_flag = 0  # {0, 1}, in test mode only 10000 gene, 1000 cells tested
if test_flag == 1:
    max_training_epochs = 10  # 3L:100, 5L:1000, 7L:1000, 9L:3000
    display_step = 1  # interval on learning curve
    snapshot_step = 5  # interval of saving session, imputation
    m = 1000
    n = 15000


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
