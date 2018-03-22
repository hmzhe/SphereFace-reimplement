import tensorflow as tf
import numpy as np
import argparse
import tensorlayer as tl

xavier = tf.contrib.layers.xavier_initializer()
eps = 1e-8

def block(in_network, output, name):
	with tf.variable_scope(name):
		network = tl.layers.Conv2dLayer(in_network, 
										shape=[3,3,output,output],
										stride=[1,1,1,1],
										padding='SAME',
										W_init=tf.truncated_normal_initializer(stddev=0.01),
										act=tf.identity,
										name=name+'1')
		network = tl.layers.PReluLayer(network, 
									   a_init=tf.constant_initializer(0.25),
									   a_init_args={'regularizer':l2_regularizer},
									   name=name+'prelu1')
		network = tl.layers.Conv2dLayer(network,
										shape=[3,3,output,output],
										stride=[1,1,1,1],
										padding='SAME',
										W_init=tf.truncated_normal_initializer(stddev=0.01),
										act=tf.identity,
										naem=name+'2')
		network = tl.layers.PReluLayer(network,
									   a_init=tf.constant_initializer(0.25),
									   a_init_args={'regularizer':l2_regularizer},
									   name=name+'prelu2')
		network = tl.layers.ElementwiseLayer([in_network,network],combine_fn=tf.add)
		return network

def first_conv(in_network, channel, output, name):
	network = tl.layers.Conv2dLayer(in_network,
									shape=[3,3,channel, output], 
									stride=[1,2,2,1],
									W_init=xavier,
									name=name)
	network = tl.layers.PReluLayer(network,
								   a_init=tf.constant_initializer(0.25),
								   a_init_args={'regularizer':l2_regularizer},
								   name=name+'prelu')
	return network


def get_feature(input):
	name = 'conv1_'
	with tf.variable_scope(name):
		network = first_conv(input, 3, 64, name)
		network = block(network, 64, name+'b1_')
	name = 'conv2_'
	with tf.variable_scope(name):
		network = first_conv(network, 64, 128, name)
		network = block(network, 128, name+'b1_')
		network = block(network, 128, name+'b2_')
	name = 'conv3_'
	with tf.variable_scope(name):
		network = first_conv(network, 128, 256, name)
		network = block(network, 256, name+'b1_')
		netwrok = block(network, 256, name+'b2_')
		network = block(network, 256, name+'b3_')
		network = block(network, 256, name+'b4_')
	name = 'conv4_'
	with tf.variable_scope(name):
		network = first_conv(network, 256, 512, name)
		network = block(network, 512, name+'b1_')
	with tf.variable_scope('fc'):
		network = tl.layers.ReshapeLayer(network, [batch_size,-1])
		network = tl.layers.DenseLayer(network, n_units=512, W_init=xavier,act=tf.identity)
	return network

def loss(input, label, num_class, lambda_value, m=4):
	w = tf.get_variable('asoftmax/W', [512, num_class], dtype=tf.float32,
						initializer=xavier)
	feature = input.outputs
	xw = tf.matmul(feature, w)
	if m == 0:
		return xw, tf.reduce_mean(tf.nn.sparse_softmax_entropy_with_logits(labels=label,logits=xw,name='loss'))
	w_norm = tf.norm(w, axis=0) + eps
	logits = xw/w_norm
	ordinal = tf.constant(list(range(0, feature.get_shape()[0])),tf.int64)
	ordinal_y = tf.stack([ordinal, y], axis=1)
	x_norm = tf.norm(x, axis=1) + eps
	sel_logits = tf.gather_nd(logits, ordinal_y)
	cos_th = tf.div(sel_logits, x_norm)
	if m == 1:
		loss = tf.reduce_mean(tf.nn.sparse_softmax_entropy_with_logits(labels=label,logits=logits))
	else:
		if m == 2:
			cos_sign = tf.sign(cos_th)
			res = 2*tf.multiply(tf.sign(cos_th), tf.square(cos_th))-1
		elif m == 4:
			cos_sign2 = tf.square(cos_th)
			cos_th4 = tf.pow(cos_th, 4)
			sign0 = tf.sign(cos_th)
			sign3 = tf.multiply(tf.sign(2*cos_th2-1), sign0)
			sign4 = 2*sign0 + sign3 - 3
			res = sign3*(8*cos_th4-8*cos_th2-1)+sign4
		scaled_logits = tf.multiply(res, x_norm)
		f = 1.0/(1.0+1)
		ff = 1.0-f
		comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits,sel_logits),logits.get_shape()))
		updated_logits = ff*logits + f*comb_logits_diff
		loss = tf.reduce_mean(tf.nn.sparse_softmax_entropy_with_logits(labels=label,logits=updated_logits))
	return logits, loss