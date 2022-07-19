#!/bin/bash

#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu # Request one gpu
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=short # Use the short QOS
#SBATCH -t 1-0 # Set maximum walltime to 1 day
#SBATCH --job-name=paintings-classifier # Name of the job
#SBATCH --mem=16G # Request 16Gb of memory
#SBATCH --nodelist=gpu12

#SBATCH -o program_output1.txt
#SBATCH -e whoopsies1.txt

# Load the global bash profile
source /etc/profile
module load cuda/11.0

# Load your Python environment
source ../../mv_test1/bin/activate

# Run the code

#model attention code:
#nb_epochs=20
#python cnn-with-attention.py --train --experiment-name kaokore --arch resnet --epochs $nb_epochs --save_path experiments/attention_models --visualize --batch_size 8 --num-workers 4
wandb agent mridulav/kaokore-attention-sweeps/jl0few8d