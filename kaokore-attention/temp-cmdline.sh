#!/bin/bash

# Run the code

#model attention code:
nb_epochs=01
python cnn-with-attention.py --experiment-name dummy --arch vgg --train --epochs $nb_epochs --save_path experiments/attention_models --visualize --checkpoint experiments/attention_models/cnn_epoch0$nb_epochs.pth --lr 1e-4 --batch_size 4 --dropout-type dropout --dropout-p 0.2 --regularizer-type l1