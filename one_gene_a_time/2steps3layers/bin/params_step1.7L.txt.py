# parameters used in step1

n_hidden_1 = 800
n_hidden_2 = 600
n_hidden_3 = 400
n_hidden_4 = 200

# todo: adjust to optimized hyper-parameters when different layers used
pIn = 0.8
pHidden = 0.5
learning_rate = 0.00003  # 0.0003 for 3-7L, 0.00003 for 9L
sd = 0.00001  # 3-7L:1e-3, 9L:1e-4
batch_size = 256
training_epochs = 3  #3L:100, 5L:1000, 7L:1000, 9L:3000
display_step = 20
snapshot_step = 1000