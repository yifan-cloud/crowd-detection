# -*- coding:utf-8 -*-
#!/usr/bin/python
# coding:utf8
##-------keras模型保存为tensorflow的二进制模型-----------
import sys
from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
 
from model import *
from data import *#导入这两个文件中的所有函数
from keras.callbacks import TensorBoard
# from keras.utils.vis_utils import plot_model #保存模型图
from keras.utils import plot_model
import cv2
import os 
from keras.models import load_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    # 将会话状态冻结为已删除的计算图,创建一个新的计算图,其中变量节点由在会话中获取其当前值的常量替换.
    # session要冻结的TensorFlow会话,keep_var_names不应冻结的变量名列表,或者无冻结图中的所有变量
    # output_names相关图输出的名称,clear_devices从图中删除设备以获得更好的可移植性
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        # 从图中删除设备以获得更好的可移植性
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        # 用相同值的常量替换图中的所有变量
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
 
output_fld = sys.path[0] + r'/tmp/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(sys.path[0], r'MobileNetv2_face.hdf5')
print(weight_file_path)
K.set_learning_phase(0)
# net_model = load_model(r'./unet_membrane_int16.hdf5',custom_objects={'dice_coef_loss':dice_coef_loss_fun(1e-5),'dice_coef_fun':dice_coef_fun})
net_model = load_model(r'./MobileNetv2_face.hdf5',{"relu6":relu6})
print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)
 
# 获得当前图
sess = K.get_session()
# 冻结图
frozen_graph = freeze_session(sess, output_names=[net_model.output.op.name])
 
from tensorflow.python.framework import graph_io
graph_io.write_graph(frozen_graph, output_fld, 'MobileNetv2.pb', as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, 'MobileNetv2.pb'))
print (K.get_uid())
