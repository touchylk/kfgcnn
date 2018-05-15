# coding: utf-8
from __future__ import division
import numpy as np
import pdb
import math
from . import data_generators
import copy
class_num =200
part_map_num = {'head':0,'legs':1,'wings':2,'back':3,'belly':4,'breast':5,'tail':6}
part_map_name = {}


net_size=[38,56]
#[head_classifier,legs_classifier,wings_classifier,back_classifier,belly_classifier,breast_classifier,tail_classifier]

#def get_label_from_voc(all_imgs, classes_count, class_mapping,bird_classes_count,bird_class_mapping):

class get_voc_label(object):
    def __init__(self,all_imgs, classes_count, class_mapping,bird_classes_count,bird_class_mapping,trainable=True):
        self.all_imgs = all_imgs
        self.classes_count = classes_count
        self.class_mapping = class_mapping
        self.bird_classes_count = bird_classes_count
        self.bird_class_mapping = bird_class_mapping
        self.max_batch = len(all_imgs)
        self.batch_index = 0
        if trainable:
            self.trainable = 'trainval'
        else:
            self.trainable = 'test'

    def get_next_batch(self):
        img = self.all_imgs[self.batch_index]
        while img['imageset']!= self.trainable:
            self.batch_index+=1
            if self.batch_index>=self.max_batch:
                self.batch_index=0
            img = self.all_imgs[self.batch_index]
        label = bird_class_mapping[img['bird_class_name']]
        boxlist =[]
        size_w = img['width']
        size_h = img['height']
        for bbox in img['bboxes']:
            outbox ={}
            outbox['name']=bbox['class']
            cor = np.zeros(4)
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1= bbox['y1']
            y2 = bbox['y2']
            h = y2-y1
            w = x2-x1
            x1/=size_w
            y1/=size_h
            h/=size_h
            w /= size_w
            cor =np.array([x1,y1,w,h])
            outbox['cor'] =cor
            boxlist.append(outbox)
        img_path = img['filepath']
        boxdict, labellist =self.match(boxlist, label)
        self.batch_index += 1
        if self.batch_index >= self.max_batch:
            self.batch_index = 0
        return img_path,boxdict,labellist


    def match(boxlist, label):
        # boxlist的内容是一个dict,name为head,legs等,cor为左上角坐标,宽和长,在0-1之间
        # label的内同是一个数
        labellist = []
        boxdict = {}
        labelnp = np.zeros(class_num + 1)
        for i in range(7):
            labellist.append(labelnp)

        if len(labellist) != 7:
            raise ValueError('SDFA')
        for box in boxlist:
            index = part_map_num[box['name']]
            labellist[index][0] = 1
            labellist[index][label] = 1
            x = box['cor'][0]
            y = box['cor'][1]
            w = box['cor'][2]
            h = box['cor'][3]
            x *= net_size[1]
            w *= net_size[1]
            y *= net_size[0]
            h *= net_size[0]
            cor_np = np.array([x, y, w, h])
            cor_np = np.expand_dims(cor_np, axis=0)
            cor_np = np.expand_dims(cor_np, axis=0)
            boxdict[box['name']] = cor_np

        return boxdict, labellist


#boxlist的内容是一个dict,name为head,legs等,cor为左上角坐标,宽和长,在0-1之间
#label的内同是一个数
def match(boxlist,label):
    labellist= []
    boxdict ={}
    labelnp = np.zeros(class_num + 1)
    for i in range(7):
        labellist.append(labelnp)

    if len(labellist)!=7:
        raise ValueError('SDFA')
    for box in boxlist:
        index=part_map_num[box['name']]
        labellist[index][0]=1
        labellist[index][label]=1
        x = box['cor'][0]
        y = box['cor'][1]
        w = box['cor'][2]
        h = box['cor'][3]
        x *= net_size[1]
        w *=net_size[1]
        y *= net_size[0]
        h *=net_size[0]
        cor_np  = np.array([x,y,w,h])
        cor_np =np.expand_dims(cor_np, axis=0)
        cor_np = np.expand_dims(cor_np, axis=0)
        boxdict[box['name']] = cor_np



    return boxdict,labellist