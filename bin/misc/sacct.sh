#!/usr/bin/bash
# see Mem usage of job submitted to slurm GPUs
sacct --format="JobID,User, State, ReqMem, AveVmSize, MaxVmSize"