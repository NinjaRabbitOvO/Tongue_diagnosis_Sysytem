
from __future__ import print_function

import keras.backend as K
import numpy as np
from keras import layers
from keras.applications.imagenet_utils import (decode_predictions,
                                               preprocess_input)
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Flatten, Input, MaxPooling2D,
                          ZeroPadding2D)
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #-------------------------------#
    #   利用1x1卷积进行通道数的下降
    #-------------------------------#
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用3x3卷积进行特征提取
    #-------------------------------#
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用1x1卷积进行通道数的上升
    #-------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #-------------------------------#
    #   利用1x1卷积进行通道数的下降
    #-------------------------------#
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用3x3卷积进行特征提取
    #-------------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用1x1卷积进行通道数的上升
    #-------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    #-------------------------------#
    #   将残差边也进行调整
    #   才可以进行连接
    #-------------------------------#
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[512,512,3],classes=1000):
    img_input = Input(shape=input_shape)
    #x = ZeroPadding2D((3, 3))(img_input)

    # 512,512,3-->512,512,64
    x = Conv2D(64, (3, 3), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), name='conv2')(x)
    x = BatchNormalization(name='bn_conv2')(x)
    x = Activation('relu')(x)
    feat1 = x
    #512,512,64-->256,256,64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)


    # 256,256,64-->128,128,128
    x = conv_block(x, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='c')

    # 128,128,128-->64,64,256
    x = conv_block(x, 3, [128, 128, 256], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='d')

    # 64,64,256 -> 32,32,512
    x = conv_block(x, 3, [256, 256, 512], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='f')





    # 1,1,2048 -> 2048 -> num_classes
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)


    return feat1, feat2, feat3, feat4, feat5



