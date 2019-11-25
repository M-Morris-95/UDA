import argparse

def GetParser():
    parser = argparse.ArgumentParser(
        description='M-Morris-95 UDA parser')

    parser.add_argument('--epochs', '-epochs',
                        type=int,
                        help='Number of epochs to train on',
                        default=5,
                        required = False)

    parser.add_argument('--Lambda', '-L',
                        type=float,
                        help='unsupervised loss weighting',
                        default=1,
                        required=False)

    parser.add_argument('--TSA', '-TSA',
                        help= 'Training sample annealing mode, options False, linear, log, exponential',
                        type=str,
                        default=False,
                        required=False)

    parser.add_argument('--usup', '-U',
                        action="store_true",
                        help='Train on unsupervised data',
                        default=False,
                        required=False)

    parser.add_argument('--split',
                        type = int,
                        default = 100,
                        required = False)

    parser.add_argument('--dataset', '-D',
                        type = str,
                        default = 'CIFAR10',
                        help = 'dataset to use, either CIFAR10 or MNIST',
                        required = False)

    parser.add_argument('--u_n_batch',
                        type = int,
                        default = 225,
                        help = 'batch size for unlabelled samples',
                        required = False)

    parser.add_argument('--n_batch',
                        type = int,
                        default = 32,
                        help = 'batch size for labelled samples',
                        required = False)

    return parser