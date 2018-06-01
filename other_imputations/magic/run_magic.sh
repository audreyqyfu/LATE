source ~/vir/bin/activate

# python -u magic.v1.6.pbmc_g5561.run3.py 1 > magic.v1.6.pbmc_g5561.run3.py.log 2>&1
# DISPLAY problem, not working


# Alternative
# 'magic3': run3 parameters, David used in MAGIC paper

#cell_rows for gtex
#file='gtex_v7.count.4tissues.msk90.csv.gz'
#outname='gtex_v7.count.4tissues.msk90.magic3.csv'

#cell_columns for pbmc
file='10x_human_pbmc_68k.G9987.csv.gz'
outname='10x_human_pbmc_68k.G9987.magic3.csv'

echo $(date)

python -u ~/vir/bin/MAGIC.py csv -d $file -o $outname \
--cell-axis=rows -p 100 -t=9 -k=12 -ka=4 \
1>./$outname.log 2>&1

echo $(date)

zip $outname