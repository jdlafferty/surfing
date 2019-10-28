
import os
from WGAN import WGAN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

class wgan_gd(WGAN):

    def build_compress(self):
        self.A = tf.placeholder(tf.float32, shape = [784, meas], name = 'measures')
        self.yy = tf.reshape(self.y_out,[1,784])
        self.comp = tf.matmul(self.yy, self.A)
        self.obs = tf.placeholder(tf.float32, shape = [1,meas], name = 'observation')
        self.LOSS = tf.reduce_sum((self.comp - self.obs)**2)

        self.grad = tf.gradients(ys = self.LOSS, xs = self.z_in)
        # add op to save and restore all variables
        self.saver = tf.train.Saver()


    def seq_adamGD(self, z_init, B, img_id, step_size, step_per_net, inter):
        model_dir = "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            128, self.z_dim)
        self.chk_dir = "checkpoints/" + model_dir + '/' +self.model_name + '/' +self.model_name +".model-"

        this_z = np.array(z_init).reshape([1,self.z_dim])
        img = self.test_data[img_id].reshape([1,784])
        comp_img = np.matmul(img, B)
        for idx in range(0,100):
            iter_num = idx * inter + inter
            filename = self.chk_dir + "%d" %iter_num
            self.saver.restore(sess, filename)
            # adam gradient descent
            m = np.zeros(shape = [1,self.z_dim])
            v = np.zeros(shape = [1,self.z_dim])
            b1 = 0.9
            b2 = 0.999
            if idx == 99:    
                step_per_net = step_per_net*5
                step_size = step_size/5  
            for i in range(0, step_per_net):
                g = self.sess.run(self.grad, feed_dict = {self.z_in: this_z, self.A: B, self.obs:comp_img})
                gradient = np.array(g).reshape([1,self.z_dim])
                # check gradient size
                if np.sqrt(np.sum(gradient**2)) < 1e-3:
                    break                    
                m = b1*m +(1-b1)*gradient
                v = b2*v +(1-b2)*gradient**2
                m_hat = m/(1-b1**(i+1))
                v_hat = v/(1-b2**(i+1))
                this_z = this_z - step_size * m_hat / (np.sqrt(v_hat) + 1e-8)
                this_z = clip(this_z)
        filename = self.chk_dir + "%d" % (inter*100)
        self.saver.restore(sess, filename)
        pred = self.sess.run(self.yy, feed_dict={self.z_in:this_z})
        error = np.sqrt(np.sum((pred - img)**2)/784)
        return error


    def normal_adamGD(self, z_init, B, img_id, step_size, steps, iter_num):
        model_dir = "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            128, self.z_dim)
        self.chk_dir = "checkpoints/" + model_dir + '/' +self.model_name + '/' +self.model_name +".model-"

        this_z = np.array(z_init).reshape([1,self.z_dim])
        img = self.test_data[img_id].reshape([1,784])
        comp_img = np.matmul(img, B)

        filename = self.chk_dir + "%d" %iter_num
        self.saver.restore(sess, filename)

        # adam gradient descent
        m = np.zeros(shape = [1,self.z_dim])
        v = np.zeros(shape = [1,self.z_dim])
        b1 = 0.9
        b2 = 0.999
        for i in range(0, steps):
            g = self.sess.run(self.grad, feed_dict = {self.z_in: this_z, self.A: B, self.obs:comp_img})
            gradient = np.array(g).reshape([1,self.z_dim])
            m = b1*m +(1-b1)*gradient
            v = b2*v +(1-b2)*gradient**2
            m_hat = m/(1-b1**(i+1))
            v_hat = v/(1-b2**(i+1))
            this_z = this_z - step_size * m_hat / (np.sqrt(v_hat) + 1e-8)
            this_z = clip(this_z)
        filename = self.chk_dir + "%d" %iter_num
        self.saver.restore(sess, filename)
        pred = self.sess.run(self.yy, feed_dict={self.z_in:this_z})
        error = np.sqrt(np.sum((pred - img)**2)/784)
        return error

def clip(x):
    return np.maximum(np.minimum(x,1),-1)

    
desc = "Random Seeds"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--num', type=int, help='number of points')
parser.add_argument('--dim', type=int, help='hidden dimension')
parser.add_argument('--meas', type=int, help='num of measures')
args = parser.parse_args()

args = parser.parse_args()
dim = args.dim
inter = 100
seed = args.seed
n = args.num
meas = args.meas

tf.reset_default_graph()
gan = wgan_gd(sess = None,
            epoch=1,
            batch_size=1,
            z_dim=dim,
            dataset_name="fashion-mnist",
            checkpoint_dir="checkpoint",
            result_dir="results",
            log_dir="logs")
gan.build_model()
gan.build_compress()

np.random.seed(seed)




ind = np.random.randint(10000,size = n)

init = np.random.randn(n,1,dim)
M = np.random.randn(n,784,meas) 
err_seq = []
err_normal = []

with tf.Session() as sess:
    gan.sess = sess
    for i in range(n):
        e_seq = gan.seq_adamGD(init[i], M[i], ind[i], 0.01, 3000, inter)
        e_normal = gan.normal_adamGD(init[i], M[i], ind[i], 0.002, 20000,inter*100)
        err_seq.append(e_seq)
        err_normal.append(e_normal)


err_seq = np.array(err_seq)
err_normal = np.array(err_normal)

if not os.path.isdir("compress/dim%d" %dim):
    os.mkdir("compress/dim%d" %dim)

err_seq.tofile("compress/dim%d/seq-m%d-%d.bin" %(dim, meas, seed))
err_normal.tofile("compress/dim%d/normal-m%d-%d.bin" %(dim, meas, seed))










