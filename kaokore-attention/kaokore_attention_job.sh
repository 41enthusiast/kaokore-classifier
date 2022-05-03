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
python cnn-with-attention.py --experiment-name kaokore_attn_v2_2 --arch vgg --train --epochs $nb_epochs --save_path experiments/attention_models --visualize --checkpoint experiments/attention_models/cnn_epoch0$nb_epochs.pth --lr 1e-4 --batch_size 8 --dropout-type drop --dropout-p 0.2 --weight-decay 0.1 --regularizer-type l1