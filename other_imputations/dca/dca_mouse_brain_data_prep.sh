# step1
# data_h5_2_csv.py

# step2
cat mouse_brain.10kg.csv | sed s/^b\'//g | sed s/\'//g | sed s/,b/,/g > mouse_brain.g10k_c1.3m.csv
cat mouse_brain.28kg.csv | sed s/^b\'//g | sed s/\'//g | sed s/,b/,/g > mouse_brain.g28k_c1.3m.csv