import json
import re

from unet_keras.nets.unet import Unet as unet
from PIL import Image
import numpy as np
import colorsys
import copy
import os
import matplotlib.pyplot as plt


class Unet(object):
    _defaults = {
        "model_path": 'logs/ep016-loss0.065-val_loss0.072.h5',
        "model_image_size": (512, 512, 3),
        "num_classes": 2,
        "blend": True,
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        self.model = unet(self.model_image_size, self.num_classes)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

        if self.num_classes == 2:
            self.colors = [(255, 255, 255), (0, 0, 0)]
        elif self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                          for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        img = [np.array(img) / 255]
        # print(img)
        img = np.asarray(img)
        pr = self.model.predict(img)[0]  # Returns: Numpy array(s) of predictions.

        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0], self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
             int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]

        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))

        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
        # 此时的image便是经过Unet网络预测的结果
        filename = "./white_xy.txt"
        for x in range(orininal_w):
            for y in range(orininal_h):
                if image.getpixel((x, y)) == (255, 255, 255):  # 取出所有白色的像素坐标
                    # print(x,y)
                    numbers = []
                    numbers.append(str(x) + "," + str(y))

                    file = open(filename, 'a+')
                    for i in range(len(numbers)):
                        file.write(str(numbers[i]) + '\n')
        file.close()
        plt.imshow(image)
        plt.show()

        if self.blend:  # 两张图片相加
            # img = old_img×0.7+image×0.3
            image = Image.blend(old_img, image, 0.05)

        # plt.imshow(image)
        # plt.show()
        x_test = []
        y_test = []

        f = open("./white_xy.txt")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            lineout = re.split('[,]+', line.strip())
            x_test.append(lineout[0])
            y_test.append(lineout[1])
            line = f.readline()
        print(x_test[0])
        print(y_test[0])
        print(len(x_test))
        for i in range(len(x_test)):
            image.putpixel((int(x_test[i]), int(y_test[i])), (255, 255, 255))
        f.close()

        with open("./white_xy.txt", 'r+') as ff:
            ff.truncate(0)
        # plt.imshow(image)
        # plt.show()
        return image
