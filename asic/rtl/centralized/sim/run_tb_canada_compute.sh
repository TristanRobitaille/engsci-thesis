#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64000M        # memory per node
#SBATCH --time=00-10:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load apptainer/1.2.4
apptainer run engsci-thesis.sif asic/rtl/centralized/sim/tb_for_canada_computer.sh