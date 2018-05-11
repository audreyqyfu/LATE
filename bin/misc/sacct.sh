#!/usr/bin/bash
# see Mem usage of job submitted to slurm GPUs
sacct --format="JobID, JobName, Partition, AllocCPUS, ReqMem, AveVmSize,
MaxVmSize, State, ExitCode"