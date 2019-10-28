import os
from VAE import VAE
from utils import show_all_variables
from utils import check_folder
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dimz', type=int, help='dim of z')

args = parser.parse_args()
dim_z = args.dimz



with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    gan = VAE(sess,
                epoch=20,
                batch_size=128,
                z_dim=dim_z,
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

    # visualize learned generator
    print(" [*] Testing finished!")



