import os
from WGAN import WGAN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import glob

class wgan_gd(WGAN):

    def seq_adamGD(self, z_init, z_true, step_size, schedule, inter):
        model_dir = "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            128, self.z_dim)
        self.chk_dir = "checkpoints/" + model_dir + '/' +self.model_name + '/' +self.model_name +".model-"
        self.saver.restore(sess, self.chk_dir + "%d" % (100*inter))
        this_z = np.array(z_init).reshape([1,self.z_dim])
        z_true = np.array(z_true).reshape([1,self.z_dim])

        y_target = self.sess.run(self.y_out, feed_dict = {self.z_in: z_true})
        schedule = np.concatenate((schedule,[60000]))
        nstep = 0
        for idx,step_per_net in enumerate(schedule):
            iter_num = idx * inter + inter
            filename = self.chk_dir + "%d" %iter_num
            self.saver.restore(sess, filename)
            # adam gradient descent
            m = np.zeros(shape = [1,self.z_dim])
            v = np.zeros(shape = [1,self.z_dim])
            b1 = 0.9
            b2 = 0.999
            for i in range(0, step_per_net):
                nstep += 1
                g = self.sess.run(self.gradient, feed_dict = {self.z_in: this_z, self.y0: y_target})
                gradient = np.array(g).reshape([1,self.z_dim])
                m = b1*m +(1-b1)*gradient
                v = b2*v +(1-b2)*gradient**2
                m_hat = m/(1-b1**(i+1))
                v_hat = v/(1-b2**(i+1))
                this_z = this_z - step_size * m_hat / (np.sqrt(v_hat) + 1e-8)
                this_z = clip(this_z)
                # check convergence size on last net
                if idx == len(schedule)-1 and dist(this_z,z_true) < 1e-2:
                    break
        return (this_z,nstep)

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
parser.add_argument('--inter', type=int, help='interval')
args = parser.parse_args()

args = parser.parse_args()
dim = args.dim
inter = args.inter
seed = args.seed
n = args.num

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

np.random.seed(seed)


true = np.random.uniform(-1,1, size = [n,1,dim])
init = np.random.uniform(-1,1, size = [n,1,dim])

grid = [4000]*2+[2000]*4+[1000]*8+[500]*16+[250]*32+[125]*37
grid = np.array(grid,dtype=float)

if dim == 5:
  schedule = np.array(np.round(grid/sum(grid)*464),dtype=int)
elif dim == 10:
  schedule = np.array(np.round(grid/sum(grid)*1227),dtype=int)
elif dim == 20:
  schedule = np.array(np.round(grid/sum(grid)*3702),dtype=int)

dist_seq = []
steps_seq = []
with tf.Session() as sess:
    gan.sess = sess
    for i in range(n):
        print(i)
        (sol,nstep) = gan.seq_adamGD(init[i], true[i], 0.01, schedule, inter)
        dist_seq.append(dist(sol, true[i]))
        steps_seq.append(nstep)

dis_seq = np.array(dist_seq)
np.save("solutions_surfing/dim%d/dist-%d" % (dim, seed), dis_seq)
step_seq = np.array(steps_seq,dtype=int)
np.save("solutions_surfing/dim%d/steps-%d" % (dim, seed), step_seq)

