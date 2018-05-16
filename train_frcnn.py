# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

cfg = config.Config()
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=5)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()
options.train_path = cfg.train_path
options.input_weight_path = cfg.input_weight_path
#options.train_path = "/home/e813/dataset/VOCdevkit_2007_trainval"
if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')
#指定voc路径

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")
#读txt格式还是voc格式.


# pass the settings from the command line, and persist them in the config object


#cfg.use_horizontal_flips = bool(options.horizontal_flips)
#cfg.use_vertical_flips = bool(options.vertical_flips)
#cfg.rot_90 = bool(options.rot_90)

#cfg.model_path = options.output_weight_path
#cfg.num_rois = int(options.num_rois)

if 0:#cfg.network == 'vgg':
	#cfg.network = 'vgg'
	from keras_frcnn import vgg as nn
elif cfg.network == 'resnet50':
	from keras_frcnn import resnet as nn
	print('use resnet50')
	#cfg.network = 'resnet50'
else:
	print('Not a valid model')
	raise ValueError
#使用restnet网络


# check if weight path was passed via command line
if options.input_weight_path: #这里已经被赋值为cfg里的值
	cfg.base_net_weights = options.input_weight_path
else:
	# set the path to weights based on backend and model
	cfg.base_net_weights = nn.get_weight_path()
#设定restore路径

all_imgs, classes_count, class_mapping,bird_class_count,bird_class_mapping = get_data(options.train_path) #get_data函数在pascalvocparser.py里变

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

cfg.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print('Training bird per class:')
pprint.pprint(bird_class_count)
print('total birds class is {}'.format(len(bird_class_count)))
#exit()

config_output_filename = cfg.config_filepath #options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(cfg,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
#get_anchor_gt l里的 输入train_imgs是在pascalvoc里生成的,包含了图片路径.bbox等.
#此函数为迭代函数,输出为 yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
#x_img为cv.numpy [np.copy(y_rpn_cls), np.copy(y_rpn_regr)]为rpn标签.
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))  #roiinput是什么,要去看看清楚

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True) #共享网络层的输出.要明确输出的size

# define the RPN, built on the base layers
num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors) ##这里应该是只是做了两层简单的卷积,没有anchor的引入,anchor的体现应在损失函数中.
#返回是一个list,包括了rpn的分类和回国的只.

classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(classes_count), trainable=True) #主要这里的nb_classes改程序的时候要主要
#这里roiinput 似乎是作为一个输入,看下面怎么弄的e






model_share = Model(img_input,shared_layers)
model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier[:2])
model_test = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

#这里似乎是三个模型,三个独立的模型.但是三个模型怎么共享权值.

try:
	print('loading weights from {}'.format(cfg.base_net_weights))
	model_rpn.load_weights(cfg.base_net_weights, by_name=True)
	model_classifier.load_weights(cfg.base_net_weights, by_name=True)
	model_test.load_weights(cfg.base_net_weights, by_name=True)
	model_share.load_weights(cfg.base_net_weights,by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

model_all.compile(optimizer='sgd', loss='mae')
model_share.compile(optimizer='sgd',loss='mae')
model_test.compile(optimizer='sgd',loss='mae')
X, Y, img_data = next(data_gen_train)
#share_output = model_share.predict_on_batch(X)
#print(type(share_output))
#print(share_output.shape)
#exit(5)

epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(epoch_length)  #不清楚这句的意思.
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:
		try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data = next(data_gen_train)

			loss_rpn = model_rpn.train_on_batch(X, Y)
			#print(type(loss_rpn))
			#print(loss_rpn)
			#exit()
			#上面三行输出<type 'list'> \n [9.0311985, 8.879578, 0.15162112]
			#说明train_on_batch里边已经做了训练.

			P_rpn = model_rpn.predict_on_batch(X)
			# print(len(P_rpn))  输出:   2
			# print(type(P_rpn[0]))   输出:  <type 'numpy.ndarray'>
			# print(P_rpn[0].shape)   输出:  (1, 38, 60, 9)
			# print(P_rpn[1].shape)   输出:  (1, 38, 60, 36)
			# exit()
			#这里说明了

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
			# P_rpn[0], P_rpn[1]为rpn输出的回归和分类的值.
			#R为boxes 和概率,经过非最大抑制之后的.
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, cfg, class_mapping)
			#print(X2)
			exit(8)
			#这里,x2为roi区域的坐标标签,Y1为roi区域的分类标签,y2为roi区域的坐标标签,坐标标签还是带label的.

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []
			
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if cfg.num_rois > 1:
				if len(pos_samples) < cfg.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)

			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
			_,_, out2 = model_test.predict_on_batch[X, X2[:, sel_samples, :]]
			print(out2.shape)
			exit(16)

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			iter_num += 1

			progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))]) #可视化显示.

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if cfg.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					if cfg.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(cfg.model_path)

				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete, 呵呵.')
