import shutil
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate kaokore dataset for deel learning')
    parser.add_argument('--ds-name', type=str, default='datasets/kaokore', metavar='S',
                        help='Name of the dataset to train and validate on (default: datasets/paintings)')

    parser.add_argument('--tag-file', type=str, default='datasets/dataset_v1.2', metavar='S',
                        help='Name of the folder with the csv and class names')