import os
from VAE import VAE
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, help='img_id')
args = parser.parse_args()
img_id = args.idx


tf.reset_default_graph()
gan = VAE(sess = None,
            epoch=1,
            batch_size=1,
            z_dim=2,
            dataset_name="fashion-mnist",
            checkpoint_dir="checkpoints",
            result_dir="results",
            log_dir="logs")
gan.build_model()

img = gan.data_X[img_id].reshape([1,28,28,1])

# Make data.
X = np.arange(-4, 4, 0.04)
Y = np.arange(-4, 4, 0.04)
X, Y = np.meshgrid(X, Y)
nx = len(X)
ny = len(Y)
    
if not os.path.isdir("ImgData/img%d" %img_id):
    os.mkdir("ImgData/img%d" %img_id)

with tf.Session() as sess:
    gan.sess = sess    
    for k in range(100,101):
        net = k*50 
        gan.saver.restore(sess, "checkpoints/VAE_fashion-mnist_128_2/VAE/VAE.model-%d" %net)
        Z = np.zeros([nx,ny])
        for i in range(0,nx):
            for j in range(0,ny):
                x = np.array([[X[i,j], Y[i,j]]])
                out = sess.run(gan.f, feed_dict={gan.z_in: x, gan.y0:img})
                Z[i,j] = out
        datafile = "ImgData/img%d/surf-%d.bin" %(img_id,net)
        Z.tofile(datafile)








