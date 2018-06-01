import os
home = os.environ['HOME']

# MODE
# step1/rand_init for pre-training on ref (step1)
# step2/rand_init for one step training (late)
# step2/load_saved for transfer learning (translate)
mode = 'late'  # pre-training, translate, late
mse_mode = 'mse_omega'  # mse_omega, mse
data_transformation = 'as_is'  # as_is/log/rpm_log/exp_rpm_log (done on H)

if mode == 'pre-training':
    # Reference Pretraining
    stage = 'step1'
    run_flag = 'rand_init'
elif mode == 'translate':
    # Translate
    stage = 'step2'  # step1/step2 (not others)
    run_flag = 'load_saved'  # rand_init/load_saved
elif mode == 'late':
    # Late
    stage = 'step2'
    run_flag = 'rand_init'
else:
    raise Exception('mode err')

# HYPER PARAMETERS
L = 3  # only a reporter, changing it can't alter the model structure
l = L//2
n_hidden_1 = 400
# n_hidden_2 = 300  # update for different depth
# n_hidden_3 = 200
# n_hidden_4 = 100 # add more after changing model structure


# TRAINING PARAMETERS
pIn = 0.8
pHidden = 0.5
reg_coef = 0.0  # reg3=1e-2, can set to 0.0

if run_flag == 'rand_init':
    learning_rate = 3e-4  # step1: 3e-4 for 3-7L, 3e-5 for 9L
elif run_flag == 'load_saved':
    learning_rate = 3e-5  # step2: 3e-5 for 3-7L, 3e-6 for 9L
sd = 1e-3  # 3-7L:1e-3, 9L:1e-4, update for different depth
batch_size = 256

max_training_epochs = int(1e3)
display_step = 50  # interval on learning curve
snapshot_step = int(5e2)  # interval of saving session, imputation

[a, b, c] = [0.7, 0.15, 0.15]  # splitting proportion: train/valid/test

patience = 5  # early stop patience epochs, just print warning, early stop not implemented yet


# BMB.MAGIC
file1 = 'saver.hd5'
name1 = 'test'
file1_orientation = 'gene_row'  # cell_row/gene_row


# # GTEx
# file1 = home+'/imputation/data/gtex/g5561/\
# gtex_v7.count.g5561.rpm.log.hd5'
# name1 = 'GTEx.G5561.RPM.LOG'
# file1_orientation = 'gene_row'  # cell_row/gene_row


# # PBMC
# file1 = home+'/audrey2/imputation/data/10x_human_pbmc_68k/filtering/rpm/\
# 10xHumanPbmc.g5561.rpmLog.hd5'
# name1 = 'PBMC.G5561.RPM.LOG'
# file1_orientation = 'gene_row'  # cell_row/gene_row


# For development usage #
seed_tf = 3
test_flag = 0  # [0, 1], in test mode only 10000 gene, 1000 cells tested
if test_flag == 1:
    max_training_epochs = 20 # 3L:100, 5L:1000, 7L:1000, 9L:3000
    display_step = 2  # interval on learning curve
    snapshot_step = 10  # interval of saving session, imputation
    m = 1000
    n = 300

# Gene list (data frame index)
# Gene list
pair_list = [
    # # MBM: Cd34, Gypa, Klf1, Sfpi1
    # [4058, 7496],
    # [8495, 12871],

    # TEST
    [2, 3],

    # # PBMC G5561 Non-Linear
    # ['ENSG00000173372',
    # 'ENSG00000087086'],
    #
    # ['ENSG00000231389',
    # 'ENSG00000090382'],
    #
    # ['ENSG00000158869',
    # 'ENSG00000090382'],
    #
    # ['ENSG00000074800',
    # 'ENSG00000019582'],
    #
    # ['ENSG00000157873',
    # 'ENSG00000169583'],
    #
    # ['ENSG00000065978',
    # 'ENSG00000139193'],
    #
    # ['ENSG00000117450',
    # 'ENSG00000133112'],
    #
    # ['ENSG00000155366',
    # 'ENSG00000167996'],


]

gene_list = [x for pair in pair_list for x in pair]



# print parameters
print('\nfile1:', file1)
print('name1:', name1)
print('data_frame1_orientation:', file1_orientation)
print('\nfile2:', file1)
print('name2:', name1)
print('data_frame2_orientation:', file1_orientation)

print()

print('Parameters:')
print('mode:', mode)
print('mse_mode:', mse_mode)
print('data_transformation:', data_transformation)
print('stage:', stage)
print('init:', run_flag)
print('test_mode:', test_flag)
print('{}L'.format(L))
# print('{} Genes'.format(n))
for l_tmp in range(1, l+1):
  print("n_hidden{}: {}".format(l_tmp, eval('n_hidden_'+str(l_tmp))))

print('learning_rate:', learning_rate)
print('reg_coef:', reg_coef)
print('batch_size:', batch_size)
print('data split: [{}/{}/{}]'.format(a,b,c) )
print('pIn:', pIn)
print('pHidden:', pHidden)
print('max_training_epochs:', max_training_epochs)
print('display_step', display_step)
print('snapshot_step', snapshot_step)
print()

