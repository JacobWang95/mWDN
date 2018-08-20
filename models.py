import numpy as np 
import tensorflow as tf 
import os
import sys
from utils import *

import pdb

lp_filter = np.load('./wave/db4/lp.npy')
hp_filter = np.load('./wave/db4/hp.npy')

def get_wave_kernel(shape):
    mat_hp = np.zeros((shape[0], shape[1]))
    mat_lp = np.zeros((shape[0], shape[1]))

    for i in range(shape[1]):
        for j in range(8):
            mat_lp[2*i-j, i] = lp_filter[j]
            mat_hp[2*i-j, i] = hp_filter[j]

    return mat_lp, mat_hp

def variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def wave_op(input, len_input, scope, is_training, l1_value, weight_decay, sim_reg=0, activation=None):
    with tf.variable_scope(scope) as sc:
        lp_mat, hp_mat = get_wave_kernel([len_input, len_input/2])
        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
        biases_lp = variable_on_cpu('biases_lp', [len_input/2],
                             tf.constant_initializer(0.0))
        biases_hp = variable_on_cpu('biases_hp', [len_input/2],
                             tf.constant_initializer(0.0))
        lp_out = tf.matmul(input, lp_weight)
        lp_out = tf.nn.bias_add(lp_out, biases_lp)

        hp_out = tf.matmul(input, hp_weight)
        hp_out = tf.nn.bias_add(hp_out, biases_hp)
        
        if not activation==None:
            hp_out = activation(hp_out)
            lp_out = activation(lp_out)

        all_out = tf_concat(1, [lp_out, hp_out])
        return lp_out, hp_out, all_out


def wave_op_conv(input, len_input, scope, is_training, l1_value, weight_decay, sim_reg=0, activation=None):
    with tf.variable_scope(scope) as sc:
        input = tf.squeeze(input)
        lp_mat, hp_mat = get_wave_kernel([len_input, len_input/2])
        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
        biases_lp = variable_on_cpu('biases_lp', [len_input/2],
                             tf.constant_initializer(0.0))
        biases_hp = variable_on_cpu('biases_hp', [len_input/2],
                             tf.constant_initializer(0.0))
        lp_out = tf.matmul(input, lp_weight)
        lp_out = tf.nn.bias_add(lp_out, biases_lp)

        hp_out = tf.matmul(input, hp_weight)
        hp_out = tf.nn.bias_add(hp_out, biases_hp)

        if not activation==None:
            hp_out = activation(hp_out)
            lp_out = activation(lp_out)

        hp_out = tf.expand_dims(hp_out, -1)
        lp_out = tf.expand_dims(lp_out, -1)

        hp_out = tf.expand_dims(hp_out, -1)
        lp_out = tf.expand_dims(lp_out, -1)

        all_out = tf_concat(-1, [lp_out, hp_out])
        return lp_out, hp_out, all_out

def res_block(input_data, num_channel_out, ker_size, scope, bn, weight_decay, is_training):
    with tf.variable_scope(scope):
        num_channel_in = input_data.get_shape()[-1].value
        conv1 = conv2d(input_data, num_channel_out, [ker_size, 1], scope='conv1', bn=bn, 
            weight_decay=weight_decay, is_training=is_training)
        if num_channel_out != num_channel_in:
            side_conv = conv2d(input_data, num_channel_out, [1,1], scope='conv_side', bn=bn,
                weight_decay=weight_decay, is_training=is_training)
        else:
            side_conv = input_data
        output = tf.add(conv1, side_conv)

    return output