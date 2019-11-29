import argparse

def GetParser():
    parser = argparse.ArgumentParser(
        description='M-Morris-95 UDA parser')

    parser.add_argument('--Epochs', '-E',
                        type=int,
                        help='Number of epochs to train on',
                        default=5,
                        required = False)

    parser.add_argument('--Lambda', '-L',
                        type=float,
                        help='unsupervised loss weighting',
                        default=1,
                        required=False)

    parser.add_argument('--TSA', '-T',
                        type=str,
                        help= 'Training sample annealing mode, options False, linear, log, exponential',
                        default=False,
                        required=False)

    parser.add_argument('--Mode', '-M',
                        type = str,
                        help='Training mode, UDA, Supervised',
                        default='Supervised',
                        required=False)

    parser.add_argument('--Split', '-S',
                        type = int,
                        help='Labelled/ unlabelled split',
                        default = 4000,
                        required = False)

    parser.add_argument('--Loss',
                        type = str,
                        help='Loss function, KL_D for KL Divergences, MSE',
                        default = 'KL_D',
                        required=False)

    parser.add_argument('--Dataset', '-D',
                        type = str,
                        default = 'CIFAR10',
                        help = 'dataset to use, either CIFAR10 or MNIST',
                        required = False)

    parser.add_argument('--U_Batch', '-U',
                        type = int,
                        default = 32,
                        help = 'batch size for unlabelled samples',
                        required = False)


    parser.add_argument('--N_Batch', '-B',
                        type = int,
                        default = 32,
                        help = 'batch size for labelled samples',
                        required = False)

    return parser