# -*- coding: utf-8 -*-
from keras import backend as K
from keras.objectives import categorical_crossentropy


if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class =1.0

epsilon = 1e-4



head_l =1
legs_l =1
wings_l =1
back_l =1
belly_l=1
breast_l=1
tail_l=1

def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'th':
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		#if里边是不是tf的,不用管
		else:
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)#这三行是smoothL1范数

			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])


	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'tf':
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
			# lambda_rpn_class在本文档开头定义的,值为1.0,等下看看y_true 是什么
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num
	#等下也要看一下y_true是什么.


def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

#[head_classifier,legs_classifier,wings_classifier,back_classifier,belly_classifier,breast_classifier,tail_classifier]
def bird_loss(num):
	def bird_loss1(bird_true, y_pred_list):
		head_loss = head_l* bird_true[0][0]*categorical_crossentropy(bird_true[0][1:],y_pred_list[0])
		legs_loss = legs_l * bird_true[1][0] * categorical_crossentropy(bird_true[1][1:], y_pred_list[1])
		wings_loss = wings_l * bird_true[2][0] * categorical_crossentropy(bird_true[2][1:], y_pred_list[2])
		back_loss = back_l * bird_true[3][0] * categorical_crossentropy(bird_true[3][1:], y_pred_list[3])
		belly_loss = belly_l * bird_true[4][0] * categorical_crossentropy(bird_true[4][1:], y_pred_list[4])
		breast_loss = breast_l * bird_true[5][0] * categorical_crossentropy(bird_true[5][1:], y_pred_list[5])
		tail_loss = tail_l * bird_true[6][0] * categorical_crossentropy(bird_true[6][1:], y_pred_list[6])
		return (head_loss+legs_loss+wings_loss+back_loss+belly_loss+breast_loss+tail_loss)/7
	return bird_loss1