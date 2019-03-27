'''
Created on Sep 21, 2016

@author: jerrik
'''

import os
import sys
import time
import numpy as np
import tensorflow as tf

import utils, nest
from TfUtils import entry_stop_gradients, mkMask, reduce_avg, masked_softmax
from Capsule_masked import Capusule

class model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, config):
        """options in this function"""
        self.config = config
        self.EX_REG_SCOPE = []

        self.on_epoch = tf.Variable(0, name='epoch_count', trainable=False)
        self.on_epoch_accu = tf.assign_add(self.on_epoch, 1) # https://www.tensorflow.org/api_docs/python/tf/assign_add

        self.build()

    def add_placeholders(self):
        # shape(b_sz, sNum, wNum)
        self.ph_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ph_input')

        # shape(bsz)
        self.ph_labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='ph_labels')

        # [b_sz]
        self.ph_wNum = tf.placeholder(shape=(None,), dtype=tf.int32, name='ph_wNum')

        ## [b_sz]
        self.ph_sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='ph_sample_weights')
        self.ph_train = tf.placeholder(dtype=tf.bool, name='ph_train')

    def create_feed_dict(self, data_batch, train):
        '''data_batch:  label_ids, snt1_matrix, snt2_matrix, snt1_len, snt2_len'''

        phs = (self.ph_input, self.ph_labels, self.ph_wNum, self.ph_sample_weights, self.ph_train)
        feed_dict = dict(zip(phs, data_batch+(train,)))
        return feed_dict

    def add_embedding(self):
        """Add embedding layer. that maps from vocabulary to vectors.
        inputs: a list of tensors each of which have a size of [batch_size, embed_size]
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        vocab_sz = max(self.config.vocab_dict.values())
        with tf.variable_scope('embedding') as scp:
            self.exclude_reg_scope(scp) ## exclude scope 추가
            if self.config.pre_trained:
                embed = utils.readEmbedding(self.config.embed_path) ## embedding 파일 조회
                embed_matrix, valid_mask = utils.mkEmbedMatrix(embed, dict(self.config.vocab_dict)) ## embedding matrix 생성
                embedding = tf.Variable(embed_matrix, 'Embedding') ## embedding 생성
                partial_update_embedding = entry_stop_gradients(embedding, tf.expand_dims(valid_mask, 1)) ## a tensor have the same value of target, but some entry will have no gradient during backprop
                embedding = tf.cond(self.on_epoch < self.config.partial_update_until_epoch,
                                    lambda: partial_update_embedding, lambda: embedding) ## https://www.tensorflow.org/api_docs/python/tf/cond
            else:
                embedding = tf.get_variable(
                  'Embedding',
                  [vocab_sz, self.config.embed_size], trainable=True) ## embedding 생성
        return embedding

    def embed_lookup(self, embedding, batch_x, dropout=None, is_train=False):
        '''

        :param embedding: shape(v_sz, emb_sz)
        :param batch_x: shape(b_sz, wNum)ß
        :return: shape(b_sz, wNum, emb_sz)
        '''
        inputs = tf.nn.embedding_lookup(embedding, batch_x)
        if dropout is not None:
            inputs = tf.layers.dropout(inputs, rate=dropout, training=is_train)
        return inputs

    def flatten_attention(self, in_x, wNum, scope=None):
        '''

        :param in_x: shape(b_sz, wtstp, emb_sz)
        :param sNum: shape(b_sz, )
        :param wNum: shape(b_sz,)
        :param scope:
        :return:
        '''
        b_sz, wtstp, _ = tf.unstack(tf.shape(in_x))
        emb_sz = int(in_x.get_shape()[-1])
        with tf.variable_scope(scope or 'encoding_attention'):

            with tf.variable_scope('snt_enc'):
                ############################################################################################### encoding layer
                if self.config.seq_encoder == 'bigru':
                    birnn_wd = self.biGRU(in_x, wNum, self.config.hidden_size, scope='biGRU')
                elif self.config.seq_encoder == 'bilstm':
                    birnn_wd = self.biLSTM(in_x, wNum, self.config.hidden_size, scope='biLSTM')
                else:
                    raise ValueError('no such encoder %s'%self.config.seq_encoder)
                ###############################################################################################

                ############################################################################################### aggragation layer
                '''shape(b_sz, dim)'''
                if self.config.attn_mode == 'avg':
                    snt_rep = reduce_avg(birnn_wd, wNum, dim=1) ## average pooling ??
                elif self.config.attn_mode == 'attn':
                    snt_rep = self.task_specific_attention(birnn_wd, wNum,
                                                           int(birnn_wd.get_shape()[-1]),
                                                           dropout=self.config.dropout,
                                                           is_train=self.ph_train, scope='attention')
                elif self.config.attn_mode == 'rout':
                    snt_rep = self.routing_masked(birnn_wd, wNum,
                                                  int(birnn_wd.get_shape()[-1]),
                                                  self.config.out_caps_num,
                                                  iter=self.config.rout_iter,
                                                  dropout=self.config.dropout,
                                                  is_train=self.ph_train, scope='attention')
                elif self.config.attn_mode == 'Rrout':
                    snt_rep = self.reverse_routing_masked(birnn_wd, wNum,
                                                          int(birnn_wd.get_shape()[-1]),
                                                          self.config.out_caps_num,
                                                          iter=self.config.rout_iter,
                                                          dropout=self.config.dropout,
                                                          is_train=self.ph_train, scope='attention')
                else:
                    raise ValueError('no such attn mode %s' % self.config.attn_mode)
                ###############################################################################################
        return snt_rep

    def build(self):
        self.add_placeholders() ## placeholder
        ############################################################################################### embedding layer
        self.embedding = self.add_embedding() ## embedding
        '''shape(b_sz, ststp, wtstp, emb_sz)'''
        in_x = self.embed_lookup(self.embedding, self.ph_input,
                                 dropout=self.config.dropout, is_train=self.ph_train) ## lookup embedding
        ###############################################################################################
        snt_reps = self.flatten_attention(in_x, self.ph_wNum, scope='enc_attn') ## encoding, aggragation layer

        ############################################################################################### predictation layer
        with tf.variable_scope('classifier'):
            logits = self.Dense(snt_reps, dropout=self.config.dropout,
                                is_train=self.ph_train, activation=tf.nn.tanh)
            opt_loss = self.add_loss_op(logits, self.ph_labels)
            train_op = self.add_train_op(opt_loss)
        self.train_op = train_op
        self.opt_loss = opt_loss
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('ce_loss', self.ce_loss)
        tf.summary.scalar('opt_loss', self.opt_loss)
        tf.summary.scalar('w_loss', self.w_loss)
        ###############################################################################################

    def Dense(self, inputs, dropout=None, is_train=False, activation=None):
        loop_input = inputs
        if self.config.dense_hidden[-1] != self.config.class_num:
            raise ValueError('last hidden layer should be %d, but get %d' %
                             (self.config.class_num,
                              self.config.dense_hidden[-1]))
        for i, hid_num in enumerate(self.config.dense_hidden):
            with tf.variable_scope('dense-layer-%d' % i):
                loop_input = tf.layers.dense(loop_input, units=hid_num)

            if i < len(self.config.dense_hidden) - 1:
                if dropout is not None:
                    loop_input = tf.layers.dropout(loop_input, rate=dropout, training=is_train)
                loop_input = activation(loop_input)

        logits = loop_input
        return logits

    def add_loss_op(self, logits, labels):
        '''

        :param logits: shape(b_sz, c_num) type(float)
        :param labels: shape(b_sz,) type(int)
        :return:
        '''

        self.prediction = tf.argmax(logits, axis=-1, output_type=labels.dtype)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, labels), tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)

        exclude_vars = nest.flatten([[v for v in tf.trainable_variables(o.name)] for o in self.EX_REG_SCOPE])
        exclude_vars_2 = [v for v in tf.trainable_variables() if '/bias:' in v.name]
        exclude_vars = exclude_vars + exclude_vars_2

        reg_var_list = [v for v in tf.trainable_variables() if v not in exclude_vars]
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_var_list])
        self.param_cnt = np.sum([np.prod(v.get_shape().as_list()) for v in reg_var_list])

        print('===' * 20)
        print('total reg parameter count: %.3f M' % (self.param_cnt / 1000000.))
        print('excluded variables from regularization')
        print([v.name for v in exclude_vars])
        print('===' * 20)

        print('regularized variables')
        print(['%s:%.3fM' % (v.name, np.prod(v.get_shape().as_list()) / 1000000.) for v in reg_var_list])
        print('===' * 20)
        '''shape(b_sz,)'''
        self.ce_loss = loss
        self.w_loss = tf.reduce_mean(tf.multiply(self.ce_loss, self.ph_sample_weights))
        reg = self.config.reg

        return self.w_loss + reg * reg_loss

    def add_train_op(self, loss):

        lr = tf.train.exponential_decay(self.config.lr, self.global_step,
                                        self.config.decay_steps,
                                        self.config.decay_rate, staircase=True)
        self.learning_rate = tf.maximum(lr, 1e-5)
        if self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config.optimizer == 'grad':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adgrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        else:
            raise ValueError('No such Optimizer: %s' % self.config.optimizer)

        gvs = optimizer.compute_gradients(loss=loss)

        capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        return train_op

    def exclude_reg_scope(self, scope):
        if scope not in self.EX_REG_SCOPE: ## exclude scope 추가
            self.EX_REG_SCOPE.append(scope)

    @staticmethod
    def biLSTM(in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):

        with tf.variable_scope(scope or 'biLSTM'):
            cell_fwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen,
                                                       dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')

            x_out = tf.concat(x_out, axis=2)
            if dropout is not None:
                x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
        return x_out

    @staticmethod
    def biGRU(in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):

        with tf.variable_scope(scope or 'biGRU'):
            cell_fwd = tf.nn.rnn_cell.GRUCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.GRUCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen,
                                                       dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')

            x_out = tf.concat(x_out, axis=2)
            if dropout is not None:
                x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
        return x_out

    @staticmethod
    def task_specific_attention(in_x, xLen, out_sz, activation_fn=tf.tanh,
                                dropout=None, is_train=False, scope=None):
        '''

        :param in_x: shape(b_sz, tstp, dim)
        :param xLen: shape(b_sz,)
        :param out_sz: scalar
        :param activation_fn: activation
        :param dropout:
        :param is_train:
        :param scope:
        :return:
        '''

        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None

        with tf.variable_scope(scope or 'attention') as scope:
            context_vector = tf.get_variable(name='context_vector', shape=[out_sz],
                                             dtype=tf.float32) ## q
            in_x_mlp = tf.layers.dense(in_x, out_sz, activation=activation_fn, name='mlp') ## h, https://www.tensorflow.org/api_docs/python/tf/layers/dense

            attn = tf.tensordot(in_x_mlp, context_vector, axes=[[2], [0]])  # shape(b_sz, tstp) ## u = q*h
            attn_normed = masked_softmax(attn, xLen) ## ai = exp(ui) / sum(exp(ui))

            attn_normed = tf.expand_dims(attn_normed, axis=-1)
            attn_ctx = tf.matmul(in_x_mlp, attn_normed, transpose_a=True)  # shape(b_sz, dim, 1) ## e = ai * hi
            attn_ctx = tf.squeeze(attn_ctx, axis=[2])   # shape(b_sz, dim)
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx

    @staticmethod
    def routing_masked(in_x, xLen, out_sz, out_caps_num, iter=3,
                                dropout=None, is_train=False, scope=None):
        '''

        :param in_x: shape(b_sz, tstp, dim)
        :param xLen: shape(b_sz,)
        :param out_sz: scalar
        :param dropout:
        :param is_train:
        :param scope:
        :return:
        '''


        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
        b_sz = tf.shape(in_x)[0]
        with tf.variable_scope(scope or 'routing'):
            attn_ctx = Capusule(out_caps_num, out_sz, iter)(in_x, xLen)   # shape(b_sz, out_caps_num, out_sz)
            attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num*out_sz])
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx

    @staticmethod
    def reverse_routing_masked(in_x, xLen, out_sz, out_caps_num, iter=3,
                       dropout=None, is_train=False, scope=None):
        '''

        :param in_x: shape(b_sz, tstp, dim)
        :param xLen: shape(b_sz,)
        :param out_sz: scalar
        :param dropout:
        :param is_train:
        :param scope:
        :return:
        '''

        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
        b_sz = tf.shape(in_x)[0]
        with tf.variable_scope(scope or 'routing'):
            '''shape(b_sz, out_caps_num, out_sz)'''
            attn_ctx = Capusule(out_caps_num, out_sz, iter)(in_x, xLen, reverse_routing=True)
            attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num * out_sz])
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx