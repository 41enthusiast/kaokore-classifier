#!/bin/bash
#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu # Request one gpu
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=short # Use the short QOS
#SBATCH -t 1-0 # Set maximum walltime to 1 day
#SBATCH --job-name=paintings-classifier # Name of the job
#SBATCH --mem=4G # Request 4Gb of memory

#SBATCH -o program_output.txt
#SBATCH -e whoopsies.txt

# Load the global bash profile
source /etc/profile

# Load your Python environment
source ../mv_test1/bin/activate

# Run the code

#dataset generation
#python gen_datasets.py --mode all 
#python gen_datasets.py --output-mode multiple --mode all 

#model training
python train.py --root kaokore --log-interval 50 --epochs 20 --batch-size 32 --arch vgg --label gender --ver-name gender_v1.1_1 --optimizer adam --lr 0.01 --experiment-name kaokore-classifier-v1 #--balance #--freeze-base

#kaokore dataset generation and analysis
#python dataset.py --root "kaokore" --label gender --version protov0.1