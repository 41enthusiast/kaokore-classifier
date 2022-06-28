#!/bin/bash

#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu # Request one gpu
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=short # Use the short QOS
#SBATCH -t 1-0 # Set maximum walltime to 1 day
#SBATCH --job-name=paintings-classifier # Name of the job
#SBATCH --mem=4G # Request 4Gb of memory

#SBATCH -o program_output1.txt
#SBATCH -e whoopsies1.txt

# Load the global bash profile
source /etc/profile

# Load your Python environment
source ../../mv_test1/bin/activate

# Run the code

#model attention code:
nb_epochs=20
python base_fast_style_transfer