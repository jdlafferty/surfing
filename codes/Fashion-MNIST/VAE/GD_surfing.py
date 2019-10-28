import os
from VAE import VAE
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dimz', type=int, help='dim of z')
parser.add_argument('--num', type=int, help='number')
parser.add_argument('--seed', type=int, help='seed')
args = parser.parse_args()
dim_z = args.dimz
n = args.num
seed = args.seed


tf.reset_default_graph()
gan = VAE(sess = None,
            epoch=1,
            batch_size=1,
            z_dim=dim_z,
            dataset_name="fashion-mnist",
            checkpoint_dir="checkpoints",
            result_dir="results",
            log_dir="logs")
gan.build_model()

def seq_adam(sess, z_init, z_true, step_size, schedule):
    gan.sess = sess
    gan.saver.restore(sess, "checkpoints/VAE_fashion-mnist_128_%d/VAE/VAE.model-5000" %dim_z)
    this_z = np.array(z_init).reshape([1,dim_z])
    z_true = np.array(z_true).reshape([1,dim_z])
    y_target = sess.run(gan.y_out, feed_dict = {gan.z_in: z_true})
    schedule = np.concatenate((schedule,[60000]))
    nstep = 0
    for idx,step_per_net in enumerate(schedule):
        iter_num = idx * 50 + 50
        filename = "checkpoints/VAE_fashion-mnist_128_%d/VAE/VAE.model-%d" %(dim_z,iter_num)
        gan.saver.restore(sess, filename)
        # adam gradient descent
        m = np.zeros(shape = [1,dim_z])
        v = np.zeros(shape = [1,dim_z])
        b1 = 0.9
        b2 = 0.999
        for i in range(0, step_per_net):
            nstep += 1
            g = sess.run(gan.gradient, feed_dict = {gan.z_in: this_z, gan.y0: y_target})
            gradient = np.array(g).reshape([1,dim_z])
            m = b1*m +(1-b1)*gradient
            v = b2*v +(1-b2)*gradient**2
            m_hat = m/(1-b1**(i+1))
            v_hat = v/(1-b2**(i+1))
            this_z = this_z - step_size * m_hat / (np.sqrt(v_hat) + 1e-8)
            # check convergence size on last net
            if idx == len(schedule)-1 and dist(this_z,z_true) < 1e-2:
                break
    return (this_z,nstep)

def dist(a,b):
    c = np.sqrt(np.sum((a-b)**2))
    return c


np.random.seed(seed)


true = np.random.randn(n,1,dim_z)
init = np.random.randn(n,1,dim_z)

grid = [4000]*2+[2000]*4+[1000]*8+[500]*16+[250]*32+[125]*37
grid = np.array(grid,dtype=float)

if dim_z == 5:
  schedule = np.array(np.round(grid/sum(grid)*737),dtype=int)
elif dim_z == 10:
  schedule = np.array(np.round(grid/sum(grid)*1330),dtype=int)
elif dim_z == 20:
  schedule = np.array(np.round(grid/sum(grid)*8215),dtype=int)

dist_seq = []
steps_seq = []
with tf.Session() as sess:
    for i in range(n):
        print(i)
        (sol,nstep) = seq_adam(sess, init[i], true[i], 0.01, schedule)
        dist_seq.append(dist(sol, true[i]))
        steps_seq.append(nstep)

dis_seq = np.array(dist_seq)
np.save("solutions_surfing/dim%d/dist-%d" % (dim_z, seed), dis_seq)
step_seq = np.array(steps_seq,dtype=int)
np.save("solutions_surfing/dim%d/steps-%d" % (dim_z, seed), step_seq)

