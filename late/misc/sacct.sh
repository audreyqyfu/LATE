#!/usr/bin/bash
# see Mem usage of job submitted to slurm GPUs
sacct --format="JobID, JobName, Partition, AllocCPUS, ReqMem, AveVmSize,
MaxVmSize, State, ExitCode"

# LOG RAM USAGE, AND MORE
sacct --format="JobID, JobName, Partition, AllocCPUS, ReqMem, AveVmSize, \
MaxVmSize, State, ExitCode" > sacct.txt
sacct --format="ALL" >> sacct.txt