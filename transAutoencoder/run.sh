python -u step1.py 1>step1.py.log 2>&1
#tensorboard --logdir=pre_train/
python -u step2.py 1>step2.py.log 2>&1
#tensorboard --logdir=re_train/
