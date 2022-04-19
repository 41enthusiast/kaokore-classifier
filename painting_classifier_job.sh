#!/bin/bash
#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu # Request one gpu
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=short # Use the short QOS
#SBATCH -t 1-0 # Set maximum walltime to 1 day
#SBATCH --job-name=paintings-classifier # Name of the job
#SBATCH --mem=4G # Request 4Gb of memory

#SBATCH -o program_output
#SBATCH -e whoopsies

# Load the global bash profile
source /etc/profile

# Load your Python environment
source ../mv_test1/bin/activate

# Run the code

#dataset generation
#python gen_datasets.py --mode all 
#python gen_datasets.py --output-mode multiple --mode all 

#model training
#python train.py --ds-name datasets/single_faces_cropped --log-interval 1 --epochs 1 --experiment-name finetuning-classifier-on-paintings-temp

#kaokore dataset generation and analysis
python3 dataset.py --root "kaokore" --label gender --version protov0