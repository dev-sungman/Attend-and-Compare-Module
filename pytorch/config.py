import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # path setting
    parser.add_argument('--log', type=str, default='', help='log message')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # hyperparameter
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # hardware setting
    parser.add_argument('--gpu_idx', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)
    
    # print log
    parser.add_argument('--print_freq', type=int, default=100)
    return parser.parse_args()
