# import tensorflow as tf
import torch
import torch.nn as nn

from logHandler import get_logger

    
# def leaky_relu(x, leak=0.1):
#     return tf.maximum(tf.minimum(0.0, leak * x), x)


# def fc(name, var_in, shape, act=tf.nn.relu, bn=False, reuse=False, is_training=True):
#     logging.getLogger(__name__).info('n: %r, shp: %10r, act: %r, bn: %r, re: %r, is_tra: %r' %(name, shape, act, bn, reuse, is_training))
#     with tf.variable_scope(name, reuse=reuse) as scope:
#         w_shape = shape
#         b_shape = shape[1:2]

#         w = tf.get_variable('w', w_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
#         b = tf.get_variable('b', b_shape, initializer=tf.constant_initializer(0))
#         h = tf.nn.bias_add(tf.matmul(var_in, w), b)

#         if bn:
#             h = tf.contrib.layers.batch_norm(h, center=True, scale=True, decay=0.9, epsilon=1e-5, is_training=is_training, scope='bn', reuse=reuse)
#         out = act(h)
#         # TODO: add tensorboard 
#         return out


# def conv2d(name, var_in, kernel, stride=1, act=tf.nn.relu, bn=False, reuse=False, is_training=True, stddev=0.2):
#     logging.getLogger(__name__).info('n: %r, k: %17r, strd: %r, act: %r, bn: %r, re: %r, is_tra: %r, std: %f' %(name, kernel, stride, act, bn, reuse, is_training, stddev))
#     with tf.variable_scope(name, reuse=reuse) as scope:
#         w_shape = kernel
#         b_shape = kernel[3:4]

#         # w = tf.get_variable('w', shape=w_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
#         w = tf.get_variable('w', shape=w_shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
#         b = tf.get_variable('b', shape=b_shape, initializer=tf.constant_initializer(0.0))

#         h = tf.nn.conv2d(var_in, w, strides=[1, stride, stride, 1], padding='SAME')
#         h = tf.nn.bias_add(h, b)

#         if bn:
#             h = tf.contrib.layers.batch_norm(h, center=True, scale=True, decay=0.9, epsilon=1e-5, is_training=is_training, scope='bn', reuse=reuse)
#         out = act(h)
#         # TODO: add tensorboard histogram of h, b, 
#         return out

# def deconv2d(name, var_in, shape, kernel, stride=1, act=tf.nn.relu, bn=False, reuse=False, is_training=True, stddev=0.02):
#     logging.getLogger(__name__).info('n: %r, shp: %17r, k: %17r, strd: %r, act: %r, bn: %r, re: %r, is_tra: %r, std: %f' %(name, shape, kernel, stride, act, bn, reuse, is_training, stddev))
#     with tf.variable_scope(name, reuse=reuse) as scope:
#         w_shape = kernel
#         b_shape = kernel[2:3]

#         batch_size = tf.shape(var_in)[0]
#         output_shape = [batch_size, shape[1], shape[2], shape[3]]

#         w = tf.get_variable('w', shape=w_shape, initializer=tf.random_normal_initializer(stddev=stddev))
#         b = tf.get_variable('b', shape=b_shape, initializer=tf.constant_initializer(0.0))

#         h = tf.nn.conv2d_transpose(var_in, w, output_shape, strides=[1, stride, stride, 1], padding='SAME')
#         h = tf.nn.bias_add(h, b)
        
#         if bn:
#             h = tf.contrib.layers.batch_norm(h, center=True, scale=True, decay=0.9, epsilon=1e-5, is_training=is_training, scope='bn', reuse=reuse)
#         out = act(h)
#         # TODO: add tensorboard
#         return out

def print_netowrk(net):
    logger = get_logger(__name__)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(net)
    logger.info('Total number of parameters: {}'.format(num_params))

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.2)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.2)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.2)
            m.bias.data.zero_()
