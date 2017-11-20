# Hyper structure #
L = 7  # only a reporter, changing it can't alter the model structure
l = L//2
n_hidden_1 = 800
n_hidden_2 = 400  # update for different depth
n_hidden_3 = 200
n_hidden_4 = -1  # -1 means no such layer


# Training parameters #
pIn = 0.9
pHidden = 0.8
learning_rate = 0.0000001  # 0.0003 for 3-7L, 0.00003 for 9L, update for different depth
sd = 0.0001  # 3-7L:1e-3, 9L:1e-4, update for different depth
batch_size = 393
training_epochs = 2000000
display_step = 200  # interval on learning curve
snapshot_step = 5000  # interval of saving session, imputation
[a, b, c] = [0.85, 0.15, 0]  # splitting proportion: train/valid/test
j_lst = [4058, 7496, 8495, 12871]  # Cd34, Gypa, Klf1, Sfpi1


# file input #
# EMT.MAGIC
# file1 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.ds_10k_10.log.hd5"  # data need imputation
# file2 = "../../../../magic/results/mouse_bone_marrow/EMT_MAGIC_9k/EMT.MAGIC.9k.B.log.hd5"  # ground truth
# name1 = '(EMT_MAGIC_B.msk90)'
# name2 = '(EMT_MAGIC_B)'

# GTEx Muscle
file1 = "../../../../data/gtex/tpm_ds_muscle/gtex_v7.tpm.ds_70k_10p_log.muscle_yes.hd5"  # input X
file2 = "../../../../data/gtex/tpm_muscle/gtex_v7.tpm.log.muscle_yes.hd5"  # ground truth M
# file1 = "../../../../data/gtex/tpm_ds/gtex_v7.tpm.ds_70k_10p_log.hd5"  # input X
# file2 = "../../../../data/gtex/tpm/gtex_v7.tpm.log.hd5"  # ground truth M
file_orientation = 'cell_row'  # cell_row/gene_row
# name1 = '(muscle_ds70k)'  # uses 20GB of RAM
# name2 = '(GTEx_muscle)'


# For development usage #
seed_tf = 3
test_flag = 0  # [0, 1], in test mode only 10000 gene, 1000 cells tested
if test_flag == 1:
    training_epochs = 10  # 3L:100, 5L:1000, 7L:1000, 9L:3000
    display_step = 1  # interval on learning curve
    snapshot_step = 5  # interval of saving session, imputation


# Print parameters

print("\n# Parameters: {}L".format(L))
print(
      "\np.learning_rate :", learning_rate,
      "\np.batch_size: ", batch_size,
      "\nepoches: ", training_epochs,
      "\npIn_holder: ", pIn,
      "\npHidden_holder: ", pHidden)
