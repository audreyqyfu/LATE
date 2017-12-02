# Hyper structure #
stage = 'step2'  # step1/step2
L = 7  # only a reporter, changing it can't alter the model structure
l = L//2
n_hidden_1 = 800
n_hidden_2 = 400  # update for different depth
n_hidden_3 = 200
# n_hidden_4 = 100 # add more after changing model structure


# Training parameters #
pIn = 0.8
pHidden = 0.5
learning_rate = 3e-4  # 3e-4 for 3-7L, 3e-5 for 9L
sd = 1e-3  # 3-7L:1e-3, 9L:1e-4, update for different depth
batch_size = 394
max_training_epochs = int(8e3)
display_step = 100  # interval on learning curve
snapshot_step = int(2e3)  # interval of saving session, imputation
[a, b, c] = [0.85, 0.15, 0]  # splitting proportion: train/valid/test
patience = 5  # early stop patience epochs, just print warning, early stop not implemented yet


# file input #
# EMT.MAGIC
# file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.ds_10k_10.log.hd5"  # data need imputation
# file2 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"  # ground truth
# name1 = '(EMT_MAGIC_B.msk90)'
# name2 = '(EMT_MAGIC_B)'

# GTEx Muscle
# file1 = "../../../../data/gtex/tpm_ds_muscle/gtex_v7.tpm.ds_70k_10p_log.muscle_yes.hd5"  # input X (cell_row)
file1 = "../../../../data/gtex/tpm_msk_tissues/gtex_v7.tpm.log.msk90.muscle_heart_skin_adipose_yes.hd5"  # input X (cell_row)
file2 = "../../../../data/gtex/tpm_tissues/gtex_v7.tpm.log.muscle_heart_skin_adipose_yes.hd5"  # ground truth M (cell_row)
# file1 = "../../../../data/gtex/tpm_ds/gtex_v7.tpm.ds_70k_10p_log.hd5"  # input X (cell_row)
# file2 = "../../../../data/gtex/tpm/gtex_v7.tpm.log.hd5"  # ground truth M (gene_row)
file1_orientation = 'cell_row'  # cell_row/gene_row
file2_orientation = 'cell_row'
name1 = '(all_to_4tissues(msk))'  # uses 20GB of RAM
name2 = '(all_to_4tissues(M))'


# For development usage #
seed_tf = 3
test_flag = 0  # [0, 1], in test mode only 10000 gene, 1000 cells tested
if test_flag == 1:
    max_training_epochs = 10  # 3L:100, 5L:1000, 7L:1000, 9L:3000
    display_step = 1  # interval on learning curve
    snapshot_step = 5  # interval of saving session, imputation
    m = 1000
    n = 15000

# Gene list
pair_list = [[4058, 7496],
            [8495, 12871],
            [2, 3],
            [205, 206]
            ]

gene_list = [4058, 7496, 8495, 12871, 2, 3, 205, 206]


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

