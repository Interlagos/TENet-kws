import argparse
import tensorflow as tf

import models

MS_PER_SECOND = 1000
TF_SESSION_CONFIG = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
    log_device_placement=False,
    device_count={"GPU": 1})


def arg_config():
    parser = argparse.ArgumentParser()

# config
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Expected sample rate of the wavs',)
    parser.add_argument('--window_size_ms', type=int, default=30,
                        help='How long each spectrogram timeslice is.',)
    parser.add_argument('--window_stride_ms', type=int, default=10,
                        help='How far to move in time between spectogram timeslices.',)
    parser.add_argument('--background_frequency', type=float, default=0.8,
                        help="How frequent the background noise should be, between 0 and 1.")
    parser.add_argument('--background_max_volume', type=float, default=0.1,
                        help="How loud the background noise should be, between 0 and 1.")
    parser.add_argument('--num_mel_bins', type=int, default=64,
                        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument('--desired_samples', type=int, default=16000,
                        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument('--lower_edge_hertz', type=float, default=80.0,)
    parser.add_argument('--upper_edge_hertz', type=float, default=7600.0,)
    parser.add_argument('--num_mfccs', type=int, default=40,)
    parser.add_argument('--timeshift_ms', type=int, default=100,
                        help="Range to randomly shift the training audio by in time.")
    parser.add_argument('--wanted_words', type=str, default='yes,no,up,down,left,right,on,off,stop,go',
                        help='Words to use (others will be added to an unknown label)',)

# dataset
    parser.add_argument('--dataset_path', type=str,
                        default='../dataset_v1', help="Where the speech training data.")
    parser.add_argument('--split_dir', type=str,
                        default='configs/v1', help="the dir of the split file.")

# training config
    parser.add_argument('--steps', type=int, default=30000,
                        help='How many training steps to run',)
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='How many items to train with at once',)
    parser.add_argument("--buffer_size", default=1000, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument("--step_save_checkpoint", default=1000, type=int)
    parser.add_argument("--step_logging", default=100, type=int)
    parser.add_argument("--max_to_keep", default=5, type=int)
    parser.add_argument("--trainable_scopes", default="", type=str)

# loss and optimizer
    parser.add_argument('--optimizer', type=str, default="adam", choices=[
                        "adam", "sgd", "mom", "rmsprop"],
                        help='Optimizer (adam, gradient_descent, momentum and rmsprop).')
    parser.add_argument('--lr_list', type=str, default='0.01,0.001,0.0001',
                        help='How large a learning rate to use when training.')
    parser.add_argument('--boundaries', type=str, default='10000,20000',
                        help='When a learning rate to change when training.')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)

# save or checkpoint
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='save/tenet12/',
                        help='Directory to write event logs and checkpoint.')
    parser.add_argument('--checkpoint_path', type=str, default=None)

# model
    parser.add_argument("--arch", default="TENet12Model",
                        choices=models._available_nets, type=str)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Percentage of data dropped',)
    parser.add_argument('--width_multiplier', type=float,
                        default=1.0, help='',)
    parser.add_argument('--num_classes', type=int, default=12, help='',)
    parser.add_argument("--scope_name", default='', type=str)

# MTENet
    parser.add_argument('--kernel_list', type=str, default=None,
                        help='multi-scale kernel sizes for MTENet')

# train or eval
    parser.add_argument("--mod", default="train", choices=[
                        "train", "eval"], type=str)
    parser.add_argument("--dataset_name", default="train", choices=[
                        "train", "valid", "test"], type=str)

    args = parser.parse_args()
    args = update_args(args)

    print(args)

    return args


def update_args(args):
    args.lr_list = [float(item) for item in args.lr_list.split(',')]
    args.boundaries = [int(item) for item in args.boundaries.split(',')]
    args.wanted_words = args.wanted_words.split(',')
    args.window_size_samples = int(
        args.window_size_ms * args.sample_rate / MS_PER_SECOND)
    args.window_stride_samples = int(
        args.window_stride_ms * args.sample_rate / MS_PER_SECOND)

    args.spectrogram_length = 1 + \
        (args.desired_samples - args.window_size_samples) // args.window_stride_samples

    args.kernel_list = [int(item) for item in args.kernel_list.split(
        ',')] if args.kernel_list else None

    return args
