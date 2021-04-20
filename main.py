from model import *
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size=16
tensorboard = TensorBoard(log_dir='./log')
# https://www.cnblogs.com/ansang/p/8583304.html
# http://www.imooc.com/article/262036
# https://stackoverflow.com/questions/55983835/valueerror-output-of-generator-should-be-a-tuple-x-y-sample-weight-or-x
# https://www.cnblogs.com/hujinzhou/p/12368926.html
# https://www.jianshu.com/p/1cf3b543afff?utm_source=oschina-app
# https://www.jianshu.com/p/d23b5994db64
data_gen_args = dict(rescale=1. / 255,rotation_range=10,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
image_generator = image_datagen.flow_from_directory(r'data_scene/train',batch_size=batch_size,shuffle=True,class_mode='binary')


model = MobileNetv2((256, 256, 3), 1, 1.0)
# model=unet_resnet_101()
model.summary()
model_checkpoint = ModelCheckpoint('MobileNetv2.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(image_generator,steps_per_epoch=90,epochs=100,callbacks=[model_checkpoint,tensorboard])

# testGene = testGenerator(r"data/membrane/test")
# model = unet_resnet_101()
# model.load_weights(r"unet_membrane_resnet101.hdf5")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult(r"data/predice",results)
