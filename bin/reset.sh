#!/bin/sh
echo 'delte the following outputs from previous run'
echo '*.csv.gz plots/ pre_train/ re_train/ step1/ step2/ slurm*'
echo 'input: yes/no'
read user_input
echo $user_input
if [ "$user_input" == "yes" ]
then
    rm -rf *csv.gz
    rm -rf plots/
    rm -rf pre_train/
    rm -rf re_train/
    rm -rf step1/
    rm -rf step2/
    rm -rf slurm*
    echo 'deletion complete'
else
    echo 'nothing deleted'
fi
