file_h = 'saver.hd5'
file_h_ori = 'gene_row'  # gene_row/cell_row

file_x = 'm.hd5'
file_x_ori = 'gene_row'

file_m = 'm.hd5'
file_m_ori = 'gene_row'

tag = 'test_tag'

data_transformation = 'as_is'  # as_is/log/rpm_log/exp_rpm_log (done on H)

# Gene list
pair_list = [
    # # MBM: Cd34, Gypa, Klf1, Sfpi1
    # [4058, 7496],
    # [8495, 12871],
    # TEST
    [2, 3],

    # PBMC G5561
    #dCov table2
    ['ENSG00000074800', 'ENSG00000019582'],
    ['ENSG00000188976', 'ENSG00000188290'],
    ['ENSG00000074800', 'ENSG00000196126'],
    #Renyi table2
    ['ENSG00000155366', 'ENSG00000090382'],
    ['ENSG00000158869', 'ENSG00000090382'],
    #Excluding zeros, RPM,Renyi
    ['ENSG00000242485', 'ENSG00000197448'],
    # Excluding zeros, RPM,dCov
    ['ENSG00000187608', 'ENSG00000019582'],

]

gene_list = [
    # # MBM
    # 4058, 7496, 8495, 12871,
    # TEST
    2, 3,
    'ENSG00000188976', 'ENSG00000188290',
    # PBMC G5561
    'ENSG00000074800', 'ENSG00000019582',
    'ENSG00000074800', 'ENSG00000196126',

    'ENSG00000155366', 'ENSG00000090382',
    'ENSG00000158869', 'ENSG00000090382',

    'ENSG00000242485', 'ENSG00000197448',

    'ENSG00000187608', 'ENSG00000019582',

]
