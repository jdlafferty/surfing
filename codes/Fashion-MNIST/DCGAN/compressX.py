import os
from GAN import GAN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import glob

class gan_gd(GAN):

    def build_compress(self):
        self.A = tf.placeholder(tf.float32, shape = [784, meas], name = 'measures')
        self.yy = tf.reshape(self.y_out,[1,784])
        self.comp = tf.matmul(self.yy, self.A)
        self.obs = tf.placeholder(tf.float32, shape = [1,meas], name = 'observation')
        self.LOSS = tf.reduce_sum((self.comp - self.obs)**2)

        self.grad = tf.gradients(ys = self.LOSS, xs = self.z_in)
        # add op to save and restore all variables
        self.saver = tf.train.Saver()


    def seq_adamGD(self, z_init, z_true, B, step_size, step_per_net, inter):
        model_dir = "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            128, self.z_dim)
        self.chk_dir = "checkpoints/" + model_dir + '/' +self.model_name + '/' +self.model_name +".model-"
        self.saver.restore(sess, self.chk_dir + "%d" % (100*inter))
        this_z = np.array(z_init).reshape([1,self.z_dim])
        z_true = np.array(z_true).reshape([1,self.z_dim])
        comp_img = self.sess.run(self.comp, feed_dict = {self.z_in: z_true, self.A:B})

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
        ddist = dist(this_z, z_true)
        return ddist


    def normal_adamGD(self, z_init, z_true, B, step_size, steps, iter_num):
        model_dir = "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            128, self.z_dim)
        self.chk_dir = "checkpoints/" + model_dir + '/' +self.model_name + '/' +self.model_name +".model-"
        this_z = np.array(z_init).reshape([1,self.z_dim])
        z_true = np.array(z_true).reshape([1,self.z_dim])
        
        filename = self.chk_dir + "%d" %iter_num
        self.saver.restore(sess, filename)
        comp_img = self.sess.run(self.comp, feed_dict = {self.z_in: z_true, self.A:B})
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
        ddist = dist(this_z, z_true)
        return ddist

    
def dist(a,b):
    c = np.sqrt(np.sum((a-b)**2))
    return c

def clip(x):
    return np.maximum(np.minimum(x,1),-1)

desc = "Random Seeds"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--num', type=int, help='number of points')
parser.add_argument('--dim', type=int, help='hidden dimension')
parser.add_argument('--meas', type=int, help='number of measures')
args = parser.parse_args()

args = parser.parse_args()
dim = args.dim
inter = 100
seed = args.seed
n = args.num
meas = args.meas

tf.reset_default_graph()
gan = gan_gd(sess = None,
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


true = np.random.uniform(-1,1, size = [n,1,dim])
init = np.random.uniform(-1,1, size = [n,1,dim])
M = np.random.randn(n,784,meas) 

dist_normal = []
dist_seq = []
with tf.Session() as sess:
    gan.sess = sess
    for i in range(n):
        sol1 = gan.seq_adamGD(init[i], true[i], M[i],0.01, 3000, inter)
        sol2 = gan.normal_adamGD(init[i], true[i], M[i], 0.002, 20000, inter*100)
        dist_seq.append(sol1)
        dist_normal.append(sol2)


dis_seq = np.array(dist_seq)
dis_normal = np.array(dist_normal)

if not os.path.isdir("compressX/dim%d" %dim):
    os.mkdir("compressX/dim%d" %dim)

dis_seq.tofile("compressX/dim%d/seq-m%d-%d.bin" %(dim, meas, seed))
dis_normal.tofile("compressX/dim%d/normal-m%d-%d.bin" %(dim, meas, seed))









