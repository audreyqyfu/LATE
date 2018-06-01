import os
home = os.environ['HOME']

fname_imputation = './step2/imputation.step2.hd5'
ori_imputation = 'cell_row'  # gene_row/cell_row
transformation_imputation = 'as_is'

# BMB.MAGIC
fname_input = home + '/imputation/data/10x_human_pbmc_68k' \
        '/filtering/10x_human_pbmc_68k.g949.hd5'
name_input = 'test_pbmc'
ori_input = 'gene_row'
transformation_input = 'log'  # as_is/log/rpm_log/exp_rpm_log

# # Mouse Brain Small
# fname_input = home + '/imputation/data/10x_mouse_brain_1.3M/20k/' \
#               'mouse_brain.10kg.h5'
# genome = 'mm10'  # only for 10x_genomics sparse matrix h5 data
# name_input = 'test_mouse_brain_sub20k'
# ori_input = 'gene_row'  # cell_row/gene_row
# transformation_input = 'log'  # as_is/log/rpm_log/exp_rpm_log

fname_ground_truth = fname_input
name_ground_truth = name_input
ori_ground_truth = ori_input  # cell_row/gene_row
transformation_ground_truth = transformation_input  # as_is/log/rpm_log/exp_rpm_log

tag = 'Eval'  # folder name

# Gene list
pair_list = [
    # # MBM: Cd34, Gypa, Klf1, Sfpi1
    # [4058, 7496],
    # [8495, 12871],
    #
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

gene_list = [
    # # MBM
    # 4058, 7496, 8495, 12871,

    # TEST
    2, 3,
    # # 'ENSG00000188976', 'ENSG00000188290',

    # # PBMC G5561 Non-Linear
    # 'ENSG00000173372',
    # 'ENSG00000087086',
    #
    # 'ENSG00000231389',
    # 'ENSG00000090382',
    #
    # 'ENSG00000158869',
    # 'ENSG00000090382',
    #
    # 'ENSG00000074800',
    # 'ENSG00000019582',
    #
    # 'ENSG00000157873',
    # 'ENSG00000169583',
    #
    # 'ENSG00000065978',
    # 'ENSG00000139193',
    #
    # 'ENSG00000117450',
    # 'ENSG00000133112',
    #
    # 'ENSG00000155366',
    # 'ENSG00000167996',

]
