# -*- coding: utf-8 -*-
import os
import cv2
import xml.etree.ElementTree as ET
import config
import numpy as np
cfg = config.Config()
def get_data(input_path):
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	bird_classes_count ={}
	bird_class_mapping = {}

	visualise = False

	data_path = '/media/e813/E/dataset/CUBbird/CUB_200_2011/CUB_200_2011'#[os.path.join(input_path,s) for s in cfg.pascal_voc_year]
	

	print('Parsing annotation files')

	if True:

		annot_path = os.path.join(data_path, 'xml')
		imgs_path = os.path.join(data_path, 'images')
		imgsets_path_trainval = os.path.join(data_path, 'train.txt')
		imgsets_path_test = os.path.join(data_path, 'test.txt')

		trainval_files = []
		test_files = []
		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.jpg')
		except Exception as e:
			print(e)

		try:
			with open(imgsets_path_test) as f:
				for line in f:
					test_files.append(line.strip() + '.jpg')
		except Exception as e:
			if data_path[-7:] == 'VOC2012':
				# this is expected, most pascal voc distibutions dont have the test.txt file
				pass
			else:
				print(e)
		
		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		idx = 0
		for annot in annots:
			if True:
				idx += 1

				et = ET.parse(annot)
				element = et.getroot()

				#element_objs = element.findall('object')
				element_parts = element.find('parts')
				element_filename = element.find('img_path').text
				#print(element_filename)
				element_width = int(element.find('size').find('width').text)
				#print(element_width)
				element_height = int(element.find('size').find('heigth').text)
				oneparts = element_parts.findall('onepart')
				bird_class_name = element.find('class_name').text
				if bird_class_name not in bird_classes_count:
					bird_classes_count[bird_class_name]= 1
				else:
					bird_classes_count[bird_class_name] += 1
				if bird_class_name not in bird_class_mapping:
					bird_class_mapping[bird_class_name] = len(bird_class_mapping)
				#bird_class_index = {}

				if len(oneparts) > 0:
					annotation_data = {'filepath': (data_path+element_filename), 'width': element_width,
									   'height': element_height, 'bboxes': []}
					element_train_or_test = element.find('train_or_test').text
					if element_train_or_test == 'train':
						annotation_data['imageset'] = 'trainval'
					elif element_train_or_test == 'test':
						annotation_data['imageset'] = 'test'
					else:
						annotation_data['imageset'] = 'trainval'
						print 'error'
						raise ValueError
				annotation_data['bird_class_name'] = bird_class_name

				for onepart in oneparts:
					class_name = onepart.find('name').text
					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)

					part_bbox = onepart.find('bndbox')
					part_x = float(part_bbox.find('x').text)
					part_y = float(part_bbox.find('y').text)
					part_width = float(part_bbox.find('width').text)
					part_heigth = float(part_bbox.find('heigth').text)
					x1 = int(round(part_x - part_width/2))
					x2 = int(round(part_x + part_width/2))
					y1 = int(round(part_y - part_heigth/2))
					y2 = int(round(part_y + part_heigth/2))
					#x1 = int(round(float(obj_bbox.find('xmin').text)))
					#y1 = int(round(float(obj_bbox.find('ymin').text)))
					#x2 = int(round(float(obj_bbox.find('xmax').text)))
					#y2 = int(round(float(obj_bbox.find('ymax').text)))
					difficulty = (0 == 1)
					annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
				all_imgs.append(annotation_data)

				#if annotation_data['imageset']=='test':
				#	visualise = True
				#else:
				#	visualise = False
				#	print annotation_data['imageset']

				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					#print(annotation_data['filepath'])
					#print(img.shape)
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
									  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					print annotation_data
					cv2.waitKey(0)

			#except Exception as e:
			#	print(e)
			#	print('oo')
			#	continue
	return all_imgs, classes_count, class_mapping,bird_classes_count,bird_class_mapping
		# all_imgs 是annotation_data的列表
		# 每一个annotationdata是一个dict,包含 了''filepath,width,height,'bboxes,imageset
		#其中,bboxes是一个列表,每一个box是一个字典
		#

