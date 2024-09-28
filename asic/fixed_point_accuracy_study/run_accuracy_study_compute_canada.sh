#!/bin/bash 
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=48
#SBATCH --mem=128GB
#SBATCH --time=00-06:59 # time (DD-HH:MM)
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --constraint=[skylake|cascade]

module load apptainer/1.2.4
apptainer run engsci-thesis.sif asic/fixed_point_accuracy_study/start_study.sh
