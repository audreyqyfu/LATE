# splat simulation: 
one type of cells, Pearson correlation 99% among cells
1000 genes, 20k cells created

# A/B/B.mask90
A: randomly chosen ~10k cells (9884), [pre-training]
B: the remaining ~10k cells (10116), [ground-truth for re-training]
B.mask90: randomly set 90% of the data in matrix B to zero, to mimic drop-out in single cell rna-seq. [re-training]

# Pre-processing
1. Splatter output   
2. Lib-size normalization (MAGIC style, re-scaled to medium lib-size) 
3. log transformation skipped
4. split into A and B, masking 90% of the data in B

# Tip:
Can use this command to read pandas.DataFrame from hd5 files:
	>import pandas as pd
	>df = pd.read_hdf('xxx.hd5') #


==============================

# Details:
# splat parameters: 

Global: 
(GENES)  (CELLS)   [SEED]  
   1000    20000        2  

23 additional parameters 

Groups: 
     [Groups]  [GROUP CELLS]  
            1          20000  

Mean: 
 (Rate)  (Shape)  
    0.3      0.6  

Library size: 
(LOCATION)     (Scale)  
        15         0.2  

Exprs outliers: 
(Probability)     (Location)        (Scale)  
         0.05              4            0.5  

Diff expr: 
[Probability]    [Down Prob]     [Location]        [Scale]  
          0.1            0.5            0.1            0.4  

BCV: 
(Common Disp)          (DoF)  
          0.1             60  

Dropout: 
 [Present]  (Midpoint)     (Shape)  
     FALSE           0          -1  

Paths: 
        [From]        [Length]          [Skew]    [Non-linear]  [Sigma Factor]  
             0             100             0.5             0.1             0.8  

#Summary
dim of matrix:  1000 20000 

[1] "zero.percent= 0.0157"

#Counts
       Cell1 Cell2 Cell3 Cell4 Cell5 Cell6 Cell7 Cell8 Cell9 Cell10
Gene1    578  1068  1205   541   993   863  1123   958  1203    766
Gene2    698   985  1017   690   936   689   821   718  1457    985
Gene3    718  1000  1246   795   861   926   930  1037  1596    764
Gene4    275   236   439   146   262   244   345   198   413    324
Gene5    893  1024  1384   812   976   819   979   992  1276    878
Gene6   2478  3649  4223  2616  2437  3373  3684  2812  5817   2622
Gene7   1026  1183  1694  1247  1525  1667  1678  1688  2458   1219
Gene8    318   410   480   322   408   523   537   505   713    413
Gene9    127   298   371   130   276   328   317   239   547    193
Gene10   349   387   700   378   319   516   672   348   633    633

#fData
       Gene BaseGeneMean OutlierFactor  GeneMean
Gene1 Gene1    1.3743081             1 1.3743081
Gene2 Gene2    1.6173530             1 1.6173530
Gene3 Gene3    1.8290566             1 1.8290566
Gene4 Gene4    0.5322259             1 0.5322259
Gene5 Gene5    1.8770306             1 1.8770306
Gene6 Gene6    5.5919199             1 5.5919199

#pData
       Cell ExpLibSize
Cell1 Cell1    2732198
Cell2 Cell2    3392134
Cell3 Cell3    4490924
Cell4 Cell4    2607558
Cell5 Cell5    3216967
Cell6 Cell6    3356751


