import os
from WGAN_GP import WGAN_GP
from utils import show_all_variables
from utils import check_folder
import tensorflow as tf
import argparse

desc = "dimension"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--dim', type=int, help='input dimension')
args = parser.parse_args()
dim = args.dim




with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    gan = WGAN_GP(sess,
                epoch=20,
                batch_size=128,
                z_dim=dim,
                dataset_name='fashion-mnist',
                checkpoint_dir='checkpoints',
                result_dir='results',
                log_dir='logs')
    if gan is None:
        raise Exception("[!] There is no option for " + args.gan_type)

    # build graph
    gan.build_model()

    # show network architecture
    show_all_variables()

    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")


