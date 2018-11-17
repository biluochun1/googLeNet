# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os, tflearn
import numpy as np

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

from lib import data_util


# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

class GoogLeNet(object):

    def __init__(self, args):

        # ctrl param
        self.img_size = args.img_size
        self.label_size = args.label_size
        self.model_name = args.model_name
        self.gpu_usage = args.gpu_usage

        # net
        tflearn.config.init_graph(gpu_memory_fraction=self.gpu_usage)
        tflearn.init_graph(seed=8888, num_cores=16, gpu_memory_fraction=1)
        network = input_data(shape=[None, self.img_size, self.img_size, 3])

        # first net
        conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)

        # second net
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

        # inception_3a
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu',
                                   name='inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_3a_5_5_reduce')
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu',
                                   name='inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu',
                                        name='inception_3a_pool_1_1')

        # merge the inception_3a__
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                    mode='concat', axis=3)

        # inception_3b
        inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu',
                                          name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu',
                                   name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu',
                                          name='inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_3b_pool_1_1')

        # merge the inception_3b_*
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                    mode='concat', axis=3, name='inception_3b_output')

        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

        # inception_4a
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu',
                                          name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu',
                                   name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu',
                                   name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4a_pool_1_1')

        # merge the inception_4a_*
        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                    mode='concat', axis=3, name='inception_4a_output')

        # inception_4b
        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu',
                                          name='inception_4b_3_3_reduce')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu',
                                   name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu',
                                          name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4b_5_5')

        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4b_pool_1_1')

        # merge the inception_4b_*
        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                    mode='concat', axis=3, name='inception_4b_output')

        # inception_4c
        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',
                                          name='inception_4c_3_3_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu',
                                   name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu',
                                          name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4c_5_5')

        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4c_pool_1_1')

        # merge the inception_4c_*
        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                    mode='concat', axis=3, name='inception_4c_output')

        # inception_4d
        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu',
                                          name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu',
                                   name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu',
                                          name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4d_pool_1_1')

        # merge the inception_4d_*
        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                    mode='concat', axis=3, name='inception_4d_output')

        # inception_4e
        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu',
                                          name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu',
                                          name='inception_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu',
                                        name='inception_4e_pool_1_1')

        # merge the inception_4e_*
        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1],
                                    axis=3, mode='concat')

        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

        # inception_5a
        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu',
                                          name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu',
                                          name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5a_pool_1_1')

        # merge the inception_5a_*
        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
                                    axis=3, mode='concat')

        # inception_5b
        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu',
                                          name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu',
                                   name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu',
                                          name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5b_5_5')
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5b_pool_1_1')

        # merge the inception_5b_*
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],
                                    axis=3, mode='concat')

        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, 0.4)

        # output stage
        loss = fully_connected(pool5_7_7, self.label_size, activation='softmax')
        # select a optimizer and loss type
        network = regression(loss, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        # save path
        model_path = 'models/%s' % self.model_name

        if not os.path.exists(model_path): os.makedirs(model_path)
        self.model = tflearn.DNN(network, checkpoint_path=model_path + '/model',
                                 max_checkpoints=3, tensorboard_verbose=2)

        # load existing model checkpoint
        ckpt = self.get_checkpoint(model_path)
        if ckpt:
            self.model.load(ckpt)
            print("load existing checkpoint from %s $$$$$$$$$$$$$$$$$$$$$$$$$$$" % ckpt)
        else:
            print("No exist model checkpoint @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    def get_checkpoint(self, model_path):
        ckpt_path = '%s/checkpoint' % model_path
        if not os.path.exists(ckpt_path):
            return
        with open(ckpt_path, 'r') as f:
            lines = [line.split(':')[1].replace('\"', '').strip() for line in f.readlines()]
            lines = lines[1:][::-1]
        for line in lines:
            path = "%s/%s" % (model_path, line)
            print(path+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if os.path.exists(path):
                print(path)
                return path
        # path = "/home/arch1tect/PycharmProjects/googleNet_classifier/models/4goods/model-7078"
        # return path

    def fit(self, X, Y, n_epoch=1000):
        print("fit data dim: X=%s, Y=%s" % (np.shape(X), np.shape(Y)))
        self.model.fit(X, Y, n_epoch=n_epoch, validation_set=0.1, shuffle=True,
                       show_metric=True, batch_size=32, snapshot_step=200,
                       snapshot_epoch=True, run_id=self.model_name)

    def predict(self, X):
        return self.model.predict(X)

    def get_data(self, dirname='17flowers', resize_pics=(227, 227), down_sampling=None):
        pkl_fnames = ["images/%s/%s" % (dirname, f) for f in os.listdir("images/%s/" % dirname) if "samples_" in f]
        if not pkl_fnames:
            pkl_fnames = data_util.image_dirs_to_samples(dirname, self.label_size,
                                                         resize=resize_pics, convert_gray=False,
                                                         filetypes=['.jpg', '.jpeg'],
                                                         down_sampling=down_sampling, categorical_Y=True,
                                                         shuffle_data=True)

        # print("[get_data]:", pkl_fnames)
        return pkl_fnames
