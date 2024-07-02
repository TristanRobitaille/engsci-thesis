#!/bin/bash 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=41
#SBATCH --mem=128GB
#SBATCH --time=00-09:59 # time (DD-HH:MM)
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID
#SBATCH --mail-user=tristan.robitaille@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load hdf5
module load boost
module load armadillo

source /home/tristanr/projects/def-xilinliu/tristanr/engsci-thesis/asic/fixed_point_accuracy_study/start_study.sh