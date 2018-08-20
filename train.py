import numpy as np 
import tensorflow as tf 
import os
import sys
from utils import *
from Utils import *
import cPickle as pickle
import argparse
import shutil
import pdb
from models import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=10000, help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--data_name', type=str, default='yoga', help='Name of UCR data [default: yoga]')
parser.add_argument('--drop_rate', type=float, default=0.1, help='Drop out rate [default: 0.1]')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay rate [default: 0.0]')
parser.add_argument('--wavelet_reg', type=float, default=0.0, help='Regularization term on the wavelet layers [default: 0.0]')
parser.add_argument('--arch', type=str, default='res', help='Deep arch used [default: resnet, options: fc, conv, res]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
MODEL = FLAGS.arch
SIM_REG = FLAGS.wavelet_reg
name_data = FLAGS.data_name

name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.system('cp %s %s' % (name_file, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

mid_neuron_num_1 = 40
dp_keep_prob = 1-(FLAGS.drop_rate)
write_result = False
weight_decay_conv = FLAGS.weight_decay
weigth_decay_fc=FLAGS.weight_decay
l1_value = 0.000
use_bn = True

DATA_ROOT = '/data/dataset/UCR_TS_Archive_2015'

x_train, y_train, x_test, y_test = load_data(DATA_ROOT, name_data)

LEN_INPUT = len(x_train[0])
num_outputs = len(np.unique(y_train))

# log_string 'conv'
log_string('Name of data: %s' %name_data)
log_string('Length of input = %d' %LEN_INPUT)
log_string('Num of output = %d' %num_outputs)


def count_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    log_string("Total training params: %.1fk" % (total_parameters / 1e3))

def count_wave_params(name):
    total_parameters = 0
    for variable in name:
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    log_string("Total mWDN params: %.1fk" % (total_parameters / 1e3))

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, BASE_LEARNING_RATE/100) # CLIP THE LEARNING RATE
    return learning_rate



def cal_loss(output, label, scope, loss_weight=1.0):
    with tf.name_scope(scope):
        soft_out = tf.nn.softmax(output)
        label_oh = tf.one_hot(label, num_outputs, on_value=1.0, off_value=0.0)
        #loss = loss_weight * tf.reduce_mean(tf.square(label_oh - soft_out))
        loss = loss_weight * tf.reduce_mean(-label_oh * tf.log(soft_out + 0.00000001))
        tf.add_to_collection('losses', loss)
    return loss, soft_out

def wave_block(input, len_input, num_outputs, is_training, scope, l1_value, dp_kp):
    with tf.variable_scope(scope):
        lp_coe, hp_coe, all_coe = wave_op(input, len_input, scope='wave_func',
            is_training=is_training, l1_value=l1_value, weight_decay=weigth_decay_fc, sim_reg=SIM_REG)

        mid_layer_1 = fully_connected(all_coe, mid_neuron_num_1, bn=use_bn, 
            is_training=is_training,  weigth_decay=weigth_decay_fc, scope='ran_min_1', bn_decay=0.0)

        mid_layer_1 = dropout(mid_layer_1, is_training, scope='dp_1', keep_prob=dp_kp)

        predict = fully_connected(mid_layer_1, num_outputs, bn=use_bn, activation_fn = None,
            is_training=is_training, weigth_decay=weigth_decay_fc, scope = 'ran_pred_1', bn_decay=0.0)
    return lp_coe, predict


def wave_block_conv(input, len_input, num_outputs, is_training, scope, l1_value, dp_kp):
    with tf.variable_scope(scope):
        lp_coe, hp_coe, all_coe = wave_op_conv(input, len_input, scope='wave_func',
            is_training=is_training, l1_value=l1_value, weight_decay=weigth_decay_fc, sim_reg=SIM_REG)

        conv1 = conv2d(all_coe, 32, [8, 1], scope='conv1', bn=use_bn, weight_decay=weight_decay_conv, 
        	is_training=is_training)
        conv1 = max_pool2d(conv1, [2,1], scope='pool1', stride=[2,1])
        conv1 = dropout(conv1, is_training, scope='dp_1', keep_prob=dp_kp)

        conv2 = conv2d(conv1, 64, [8, 1], scope='conv2', bn=use_bn, weight_decay=weight_decay_conv, 
        	is_training=is_training)
        conv2 = max_pool2d(conv2, [2,1], scope='pool2', stride=[2,1])
        conv2 = dropout(conv2, is_training, scope='dp_1', keep_prob=dp_kp)        

        conv3 = conv2d(conv2, 64, [8, 1], scope='conv3', bn=use_bn, weight_decay=weight_decay_conv, 
        	is_training=is_training)
        len_conv = conv3.get_shape()[1].value
        conv3 = avg_pool2d(conv3, [len_conv,1], scope='pool3', stride=[1,1])
        conv3 = dropout(conv3, is_training, scope='dp_1', keep_prob=dp_kp)
        conv3 = tf.squeeze(conv3)
        predict = fully_connected(conv3, num_outputs, bn=False, activation_fn=None, is_training=is_training,
            weigth_decay=0, scope='fin_pre')
    return lp_coe, predict



def wave_block_res(input, len_input, num_outputs, is_training, scope, l1_value, dp_kp):
    with tf.variable_scope(scope):
        lp_coe, hp_coe, all_coe = wave_op_conv(input, len_input, scope='wave_func',
            is_training=is_training, l1_value=l1_value, weight_decay=weigth_decay_fc, sim_reg=SIM_REG)

        conv1 = res_block(all_coe, 16, 8, scope='conv1_1', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv1 = res_block(conv1, 16, 8, scope='conv1_2', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv1 = res_block(conv1, 16, 8, scope='conv1_3', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv1 = max_pool2d(conv1, [2,1], scope='pool1', stride=[2,1])

        conv2 = res_block(conv1, 32, 5, scope='conv2_1', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv2 = res_block(conv2, 32, 5, scope='conv2_2', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv2 = res_block(conv2, 32, 5, scope='conv2_3', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv2 = max_pool2d(conv2, [2,1], scope='pool2', stride=[2,1])

        conv3 = res_block(conv2, 64, 3, scope='conv3_1', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv3 = res_block(conv3, 64, 3, scope='conv3_2', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)
        conv3 = res_block(conv3, 64, 3, scope='conv3_3', bn=use_bn, weight_decay=weight_decay_conv,
            is_training=is_training)

        len_conv = conv3.get_shape()[1].value
        conv3 = avg_pool2d(conv3, [len_conv,1], scope='pool3', stride=[1,1])
        conv3 = tf.squeeze(conv3)
        predict = fully_connected(conv3, num_outputs, bn=False, activation_fn=None, is_training=is_training,
            weigth_decay=0, scope='fin_pre')
    return lp_coe, predict


def get_model_fc(input, len_input, num_outputs, is_training, level_wave=3):
    
    lp_1, predict_1 = wave_block(input, len_input, num_outputs, is_training, 
        scope='wave_level_1', l1_value=l1_value, dp_kp=0.9)
    #predict_1 = predict_1 / tf.reduce_sum(predict_1)

    lp_2, predict_2 = wave_block(lp_1, len_input/2, num_outputs, is_training,
        scope='wave_level_2', l1_value=l1_value, dp_kp=0.8)
    #predict_2 = predict_2 / tf.reduce_sum(predict_2)
    predict_2 = tf.add(predict_2, predict_1, name='adding_2')

    lp_3, predict_3 = wave_block(lp_2, len_input/4, num_outputs, is_training,
        scope='wave_level_3', l1_value=l1_value, dp_kp=0.7)
    #predict_3 = predict_3 / tf.reduce_sum(predict_3)
    predict_3 = tf.add(predict_3, predict_2, name='adding_3')
    

    lp_4, predict_4 = wave_block(lp_3, len_input/8, num_outputs, is_training,
        scope='wave_level_4', l1_value=l1_value, dp_kp=0.6)
    predict_4 = tf.add(predict_4, predict_3, name='adding_3')

    return predict_1, predict_2, predict_3, predict_4



def get_model_conv(input, len_input, num_outputs, is_training, level_wave=3):
    
    lp_1, predict_1 = wave_block_conv(input, len_input, num_outputs, is_training, 
        scope='wave_level_1', l1_value=l1_value, dp_kp=dp_keep_prob)

    lp_2, predict_2 = wave_block_conv(lp_1, len_input/2, num_outputs, is_training,
        scope='wave_level_2', l1_value=l1_value, dp_kp=dp_keep_prob)

    predict_2 = tf.add(predict_2, predict_1, name='adding_2')
    
    lp_3, predict_3 = wave_block_conv(lp_2, len_input/4, num_outputs, is_training,
        scope='wave_level_3', l1_value=l1_value, dp_kp=dp_keep_prob)

    predict_3 = tf.add(predict_3, predict_2, name='adding_3')

    lp_4, predict_4 = wave_block_conv(lp_3, len_input/8, num_outputs, is_training,
        scope='wave_level_4', l1_value=l1_value, dp_kp=dp_keep_prob)
    predict_4 = tf.add(predict_4, predict_3, name='adding_4')

    return predict_1, predict_2, predict_3, predict_4



def get_model_res(input, len_input, num_outputs, is_training, level_wave=3):
    
    lp_1, predict_1 = wave_block_res(input, len_input, num_outputs, is_training, 
        scope='wave_level_1', l1_value=l1_value, dp_kp=dp_keep_prob)

    lp_2, predict_2 = wave_block_res(lp_1, len_input/2, num_outputs, is_training,
        scope='wave_level_2', l1_value=l1_value, dp_kp=dp_keep_prob)
    predict_2 = tf.add(predict_2, predict_1, name='adding_2')

    lp_3, predict_3 = wave_block_res(lp_2, len_input/4, num_outputs, is_training,
        scope='wave_level_3', l1_value=l1_value, dp_kp=dp_keep_prob)
    predict_3 = tf.add(predict_3, predict_2, name='adding_3')

    lp_4, predict_4 = wave_block_res(lp_3, len_input/8, num_outputs, is_training,
        scope='wave_level_4', l1_value=l1_value, dp_kp=dp_keep_prob)
    predict_4 = tf.add(predict_4, predict_3, name='adding_4')

    return predict_1, predict_2, predict_3, predict_4


def main():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=())
            input_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, LEN_INPUT))
            gt_pl = tf.placeholder(tf.int64, shape=(BATCH_SIZE))
            len_input = LEN_INPUT
            if MODEL == 'res':
                get_model = get_model_res
            elif MODEL == 'conv':
                get_model = get_model_conv
            else:
                get_model = get_model_fc
            predict_1, predict_2, predict_3, predict_4 = get_model(
                input_data, len_input, num_outputs, is_training_pl, level_wave=3)
            loss_1, pred_1 = cal_loss(predict_1, gt_pl, 'loss_1', loss_weight=1.0)
            loss_2, pred_2 = cal_loss(predict_2, gt_pl, 'loss_2', loss_weight=1.0)
            loss_3, pred_3 = cal_loss(predict_3, gt_pl, 'loss_3', loss_weight=1.0)
            loss_4, pred_4 = cal_loss(predict_4, gt_pl, 'loss_4', loss_weight=5.0)

            tf.summary.scalar('loss_1', loss_1)
            tf.summary.scalar('loss_2', loss_2)
            tf.summary.scalar('loss_3', loss_3)
            tf.summary.scalar('loss_4', loss_4)
            batch = tf.Variable(0, trainable=False)
            loss_all = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.summary.scalar('loss_all', loss_all)

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)

            if OPTIMIZER == 'momentum':
                optimizer_1 = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
                optimizer_2 = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)  
            elif OPTIMIZER == 'adam':
                optimizer_1 = tf.train.AdamOptimizer(learning_rate)
                optimizer_2 = tf.train.AdamOptimizer(learning_rate)
            else:
                raise NotImplementedError
                      
            var_list_wave = [t for t in tf.trainable_variables() if t.name.split('/')[1] == 'wave_func']
            var_list_rand = [t for t in tf.trainable_variables() if not t.name.split('/')[1] == 'wave_func']

            train_op_1 = optimizer_1.minimize(loss_all, global_step=batch, var_list = var_list_rand)
            train_op_2 = optimizer_2.minimize(loss_all, global_step=batch, var_list = var_list_wave)
            train_op = tf.group(train_op_1, train_op_2)

            correct_prediction = tf.equal(tf.argmax(pred_4, 1), gt_pl)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        count_trainable_params()
        count_wave_params(var_list_wave)


        file_size = x_train.shape[0]
        num_batches = file_size/BATCH_SIZE

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        init = tf.global_variables_initializer()

        sess.run(init)
        count = 0

        max_acc = 0
        min_loss = np.inf
        for epoch_idx in range(MAX_EPOCH):
            current_data, current_label, _ = shuffle_data(x_train, y_train)
            for batch_idx in range(num_batches):
            	count += 1
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                feed_data = current_data[start_idx:end_idx, ...]
                feed_label = current_label[start_idx:end_idx, ...]

                summary, step, _, current_loss, out, train_acc = sess.run(
                    [merged, batch, train_op, loss_all, pred_4, accuracy], 
                    feed_dict={input_data: feed_data,
                	gt_pl: feed_label,
                	is_training_pl: True})
            train_writer.add_summary(summary, step)
            if epoch_idx % 50 == 0:
                log_string("Loss for Iter %d: %f, acc = %f" %(count,current_loss,train_acc))

            if epoch_idx % 1 == 0:
                current_data, current_label, _ = shuffle_data(x_test, y_test)
                iter_test = int(len(x_test) / BATCH_SIZE)
                acc_sum = 0
                loss_sum = 0
                for test_idx in range(iter_test):
                    start_idx = test_idx * BATCH_SIZE
                    end_idx = (test_idx+1) * BATCH_SIZE
                    feed_data = current_data[start_idx:end_idx, ...]
                    feed_label = current_label[start_idx:end_idx, ...]
                    test_acc, result_t, test_loss, current_lr= sess.run(
                        [accuracy, pred_4, loss_all, learning_rate], feed_dict={
                        input_data: feed_data, 
                        gt_pl: feed_label, 
                        is_training_pl: False})

                    acc_sum += test_acc
                    loss_sum += test_loss

                acc_mean = acc_sum / float(iter_test)
                loss_mean = loss_sum / float(iter_test)
                if loss_mean <= min_loss:
                    min_loss = loss_mean
                    if write_result == True:
                        save_path = saver.save(sess, os.path.join(LOG_DIR, "model_loss.ckpt"))
                        log_string("Lowest loss model saved in file: %s" % save_path)
                if acc_mean >= max_acc: 
                    max_acc = acc_mean
                    if write_result == True:
                        save_path = saver.save(sess, os.path.join(LOG_DIR, "model_acc.ckpt"))
                        log_string("Best acc model saved in file: %s" % save_path)
            if epoch_idx % 200 == 0:    
                log_string("--------------------------------" )
                log_string("Test loss for epoch %d: %f, Acc is %f, Max Acc is %f" %(epoch_idx, loss_mean, acc_mean, max_acc))
                log_string("Current learning rate is: %f" %(current_lr))



if __name__ == "__main__":
    main()
    LOG_FOUT.close()
