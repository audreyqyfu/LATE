# Parameters #
L = 7
l = L//2
n_hidden_1 = 800
n_hidden_2 = 400  # update for different depth
n_hidden_3 = 200
n_hidden_4 = -1

pIn = 0.8
pHidden = 0.5
learning_rate = 0.0003  # 0.0003 for 3-7L, 0.00003 for 9L, update for different depth
sd = 0.0001  # 3-7L:1e-3, 9L:1e-4, update for different depth
batch_size = 256
training_epochs = 10  #3L:100, 5L:1000, 7L:1000, 9L:3000
display_step = 1  # interval on learning curve
snapshot_step = 5  # interval of saving session, imputation
