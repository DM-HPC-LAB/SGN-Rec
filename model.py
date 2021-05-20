"""
modified from https://arxiv.org/abs/1811.00855
SR-GNN: Session-based Recommendation with graph neural networks
"""

import tensorflow as tf
import math
import numpy as np
tf.set_random_seed(42)
np.random.seed(42)



class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32) 
        self.item = tf.placeholder(dtype=tf.int32)
        self.tar = tf.placeholder(dtype=tf.int32)
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

    def forward(self, re_embedding, train=True):
        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        last_h = tf.reshape(last_h, [-1, self.out_size])
        b = self.weights['embedding'][1:]
        logits = tf.matmul(last_h, b, transpose_b=True)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in,
                                                 self.adj_out: adj_out, self.alias: alias, self.mask: mask})


class SGNREC(Model):
    def __init__(self,hidden_size=100, out_size=100, batch_size=100, n_node=None,
                 lr=None, l2=None, layers=1, decay=None, lr_dc=0.1):
        super(SGNREC,self).__init__(hidden_size, out_size, batch_size)
        
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.layers = layers
        self.weights = self._init_weights()
        
        with tf.variable_scope('sgn_model', reuse=None):
            self.loss_train, _ = self.forward(self.sgc())
            

        with tf.variable_scope('sgn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(self.sgc(), train=False)
            
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _init_weights(self):
        all_weigths = dict()
        initializer = tf.random_uniform_initializer(-self.stdv, self.stdv)
        
        all_weigths['embedding'] = tf.Variable(initializer([self.n_node, self.hidden_size]), name='embedding')
        all_weigths['W_1'] = tf.Variable(initializer([2*self.out_size, 2*self.out_size]), name='W_1')
        all_weigths['W_2'] = tf.Variable(initializer([2*self.out_size, self.out_size]), name='W_2')
            
        return all_weigths

    def sgc(self):
        fin_state = tf.nn.embedding_lookup(self.weights['embedding'], self.item)

        fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
        fin_state_in = fin_state
        fin_state_out = fin_state
        adj_in = tf.pow(self.adj_in, self.layers)
        adj_out = tf.pow(self.adj_out, self.layers)
        with tf.variable_scope('sgc'):

            fin_state_in = tf.matmul(adj_in, fin_state_in)
            fin_state_out = tf.matmul(adj_out, fin_state_out)
            av = tf.concat([fin_state_in, fin_state_out], axis = -1)
            av = tf.nn.relu(tf.matmul(av, self.weights['W_1']))
            av = tf.matmul(av, self.weights['W_2'])
            
        return tf.reshape(av, [self.batch_size, -1, self.out_size])
    