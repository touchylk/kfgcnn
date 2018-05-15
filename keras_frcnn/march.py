# coding: utf-8
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