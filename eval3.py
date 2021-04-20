# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import torch
import numpy as np
import cv2
from src.crowd_count import CrowdCounter
from src import network

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
from src.utils import show_density_map
from model import *
from src.data_loader_video import ImageDataLoaderVideo
from src.data_loader_test import ImageDataLoaderTest
from keras.models import load_model
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

from tensorflow_yolo.core import utils as u
from PIL import Image
from AIDetector_pytorch import Detector
from tracker import plot_bboxes
import time
from scipy.io import loadmat

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = r"E:\model\tensorflow_yolo\yolov3_coco.pb"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()

def predict():
    # model=load_model("DenseUNet_s_extend_epoch110.hdf5",custom_objects={'focal_loss_fixed':focal_loss(gamma=2., alpha=.25)})
    # model=load_model("vnet_membrane_int16_dice_epoch110-num15_mix.hdf5",custom_objects={'generalized_dice_loss':generalized_dice_loss_fun(1e-5),'generalized_dice_coef_fun':generalized_dice_coef_fun})
    # model=load_model(r'./MobileNetv2.hdf5',{"relu6":relu6})
    # model=load_model("vnet_membrane_int16_dice.hdf5",custom_objects={'dice_coef_loss':dice_coef_loss_fun(1e-5),'dice_coef_fun':dice_coef_fun})
    # model=load_model("unet_membrane_int16.hdf5",custom_objects={'dice_coef_loss':dice_coef_loss_fun(1e-5),'dice_coef_fun':dice_coef_fun})
    data_path =  r'E:\model\mall_dataset\frames'
    matpath = 'E:\model\mall_dataset\mall_gt.mat'
    ground_truth_all = loadmat(matpath)
    data_loader = ImageDataLoaderTest(data_path)
    net = pre_MCNN()
    # print("load model successfully")

    yolo_detecter = Detector()
    precision = 0
    for j,blob in enumerate(data_loader):
        img = blob['data']
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #img=cv2.imread("result.png",1)
        # img1=cv2.resize(img,(256,256),0)
        # img1=img1.astype(np.float32)
        # img1=img1/255.0
        # img1=np.expand_dims(img1,axis=0)
        # pre=model.predict(img1)
        # print("...........")
        # print("dense is ",pre[0][0])
        #
        # if(pre[0][0] < 0.5):
        #     if hasattr(torch.cuda, 'empty_cache'):
        #         torch.cuda.empty_cache()
        #     MCNN(net,img)
        # else:
        #     time1 = time.time()
        try:
            count = int(MCNN(net,img))
        except:
            print("cannot read")
            continue
        #     time2 = time.time()
        # count = len(bboxes)
        # for i, bbox in enumerate(bboxes):
        #         # cv2.imwrite('./image/{}.jpg'.format(i), img[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        #         if hasattr(torch.cuda, 'empty_cache'):
        #             torch.cuda.empty_cache()
        #         img_,bboxes_ = yolo_detecter.detect(img[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        #         cv2.imwrite('./image/{}.jpg'.format(i), img[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        #         count += max(0,len(bboxes_)-1)
                # print('add is:',max(0,len(bboxes_)-1))

        # print("speed is: ", 1 / (time2 - time1))
        # det_img = plot_bboxes(det_img,bboxes)
        ground_truth = ground_truth_all["count"][j]

            # img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # detected_image = yolo.detect_image(img)
            # detected_image = cv2.cvtColor(np.asarray(detected_image),cv2.COLOR_RGB2BGR)
        precision += (1 - abs(ground_truth - count)/ground_truth)

        precision_now = precision/(j+1)
        print("the number of persons is {} and the ground truth is {} precision is {}".format(count,ground_truth,precision_now))


        # cv2.imshow("detected",det_img)
        # cv2.waitKey(1)

    precision /= len(ground_truth_all['count'])
    print('precision is {}'.format(precision))




def pre_MCNN():

    MCNN_model_path = './saved_models/mcnn_shtechA_58.h5' #MCNN模型路径
    net = CrowdCounter()
    trained_model = os.path.join(MCNN_model_path)
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    return net

def img_MCNN(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = int((ht/4)*4)
    wd_1 = int((wd/4)*4)
    img = cv2.resize(img,(wd_1,ht_1))
    img = img.reshape((1,1,img.shape[0],img.shape[1]))
    return img

def MCNN(net,img):
    img = img_MCNN(img)
    time1 = time.time()
    density_map = net(img)
    time2 = time.time()
    print("speed is: ", 1 / (time2 - time1))
    density_map = density_map.data.cpu().numpy()
    show_density_map(density_map)
    et_count = np.sum(density_map)
    print(et_count)
    return et_count

def yolo_detect(img):

    original_image = img
    return_tensors = u.read_pb_return_tensors(graph, pb_file, return_elements)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = u.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    with tf.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
            feed_dict={return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = u.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = u.nms(bboxes, 0.45, method='nms')
    for i,bbox in enumerate(bboxes):
        cv2.imwrite('./image/{}.jpg'.format(i),img[bbox[0]:bbox[2],bbox[1]:bbox[3]])
        print('writted')
    image = u.draw_bbox(original_image, bboxes)

    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    cv2.waitKey(1)




if __name__=="__main__":
    predict()
