from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import json
import glob
import os

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
# 导入my_win.py中内容
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from unet_keras.unet import *
from unet_keras.seg_tongue_work import *
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from keras import layers
import os
import json
import glob

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import pylab
from unet_keras.yolov4_pytorch_master.yolo import YOLO

yolo = YOLO()
from unet_keras.yolov4_pytorch_master.yolo import YOLO
from unet_keras.yolov4_pytorch_master import predict
from unet_keras.tongue_diagnosis_design.tongue_diagnosis_color.tongue_color_model  import resnet50
from unet_keras.esophagus_cancer_classification.esophagus_classification_model import resnet50
from unet_keras.yolov4_pytorch_master.yolo import *
# 创建mainWin类并传入Ui_MainWindow
import os
from unet_keras.seg_tongue_work import Ui_MainWindow
from unet_keras.home_page import Ui_HomePageWindow
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class mainWin(QMainWindow, Ui_HomePageWindow):
    def __init__(self,parent=None):
        super(mainWin, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.login)
    def login(self):
        main_win.close()
        second_main.show()
        #second_main.move((QApplication.desktop().width() - main_win.width()) / 2,(QApplication.desktop().height() - main_win.height()) / 13)

class secondmain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(secondmain,self).__init__(parent)
        self.setupUi(self)



        self.loadtongue.clicked.connect(self.openimage)
        self.loadtongue.setFlat(True)
        self.segtongue.clicked.connect(self.seg_oput)
        self.tongue_colorButton.clicked.connect(self.tongue_color_predict)
        self.tai_colorButton.clicked.connect(self.tai_color_predict)
        self.tongue_shapeButton.clicked.connect(self.tongue_shape_predict)
        self.tongue_shapeButton2.clicked.connect(self.tongur_shape_crack_predict)
        self.tongue_shapeButton3.clicked.connect(self.tongue_shape_intented_predict)
        self.tongue_recoButton.clicked.connect(self.tongue_reco)
        self.suggestionBtn.clicked.connect(self.suggestion_function)
        self.esophagus_btn.clicked.connect(self.esophagus_predict)
        self.back_btn.clicked.connect(self.back_home)
    def openimage(self):
        self.tongue_color.setText("")
        self.tai_color.setText("")
        self.tongue_shape.setText("")
        self.tongue_shape2.setText("")
        self.tongue_shape3.setText("")
        self.sugtext.setText("")
        self.esophagus_sug.setText("")
        #self.tongue_recognition.setPixmap("")
        self.tongue_recognition.setPixmap(QPixmap('./img/back.jpg').scaled(self.tongue_recognition.width(),self.tongue_recognition.height()))
        #self.segoutput.setPixmap("")
        self.segoutput.setPixmap(QPixmap('./img/back.jpg').scaled(self.segoutput.width(), self.segoutput.height()))
        imgName,imgType = QFileDialog.getOpenFileName(self,"打开图片", "img", "*.jpg;*.tif;*.png;;All Files(*)")
        if imgName == "":
            return 0
        jpg = QPixmap(imgName).scaled(self.uesr_tongue.width(),self.uesr_tongue.height())
        #jpg = QPixmap(imgName)
        jpg.save('./loadbutton_image/user_load.jpg')
        self.uesr_tongue.setPixmap(jpg)
    def tongue_reco(self):
        im_height = 224
        im_width = 224
        num_classes = 2
        img_path = "./loadbutton_image/user_load.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        #    resize image to 224x224
        img = img.resize((im_width, im_height))
        plt.imshow(img)
        # scaling pixel value to (0-1)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))

        # read class_indict
        json_path = './tongue_recognition/class_indices_tongue_recognition.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        json_file = open(json_path, "r")
        class_indict = json.load(json_file)
        feature = resnet50(num_classes=2, include_top=False)
        feature.trainable = False
        model = tf.keras.Sequential([feature,
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(1024, activation="relu"),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(num_classes),
                                     tf.keras.layers.Softmax()])
        weights_path = './tongue_recognition/save_weights_tongue_recognition/resNet_50.ckpt'
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
        model.load_weights(weights_path)
        # prediction
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        if class_indict[str(predict_class)] == 'tongue':
            img1 = './loadbutton_image/user_load.jpg'
            r_img = predict.predict_oput(img1)
            #r_img = yolo.detect_image(img1)
            # r_image.show()
            self.tongue_recognition.setPixmap(QPixmap('./tonguerecbtn_image/user_tonguerec.jpg').scaled(self.tongue_recognition.width(), self.tongue_recognition.height()))
        else :
            #self.tongue_recoButton.clicked.connect(self.showDialog)
            QMessageBox.warning(self,"消息提示框","请上传正确的舌象",QMessageBox.Yes | QMessageBox.No)
    def seg_oput(self):
        img =Image.open('./tonguerecbtn_image/user_tonguerec.jpg')
        unet = Unet()
        r_image = unet.detect_image(img)
        r_image.save('./img/jieguo.jpg')
        self.segoutput.setPixmap(QPixmap('./img/jieguo.jpg').scaled(self.segoutput.width(),self.segoutput.height()))
        print("预测结束")
    def tongue_color_predict(self):
        im_height = 224
        im_width = 224
        num_classes = 5
        # load image
        img_path = "./img/jieguo.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        #    resize image to 224x224
        img = img.resize((im_width, im_height))
        # scaling pixel value to (0-1)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))
        # read class_indict
        json_path = './tongue_diagnosis_design/tongue_diagnosis_color/class_indices_tongue_color.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        json_file = open(json_path, "r")
        class_indict = json.load(json_file)
        feature = resnet50(num_classes=5, include_top=False)
        feature.trainable = False
        model = tf.keras.Sequential([feature,
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(1024, activation="relu"),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(num_classes),
                                     tf.keras.layers.Softmax()])

        weights_path = './tongue_diagnosis_design/tongue_diagnosis_color/save_weights_tongue_color/resNet_50.ckpt'
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
        model.load_weights(weights_path)
        # prediction
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)
        print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        #plt.title(print_res)
        #print(print_res)
        #plt.show()
        self.tongue_color.setText( " {} ".format(class_indict[str(predict_class)]))
    def tai_color_predict(self):
        im_height = 224
        im_width = 224
        num_classes = 3
        img_path = "./img/jieguo.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        #    resize image to 224x224
        img = img.resize((im_width, im_height))
        # scaling pixel value to (0-1)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))

        # read class_indict
        json_path = './tongue_diagnosis_design/tongue_diagnosis_tai_color/class_indices_tai_color.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        feature = resnet50(num_classes=3, include_top=False)
        feature.trainable = False
        model = tf.keras.Sequential([feature,
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(1024, activation="relu"),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(num_classes),
                                     tf.keras.layers.Softmax()])

        weights_path = './tongue_diagnosis_design/tongue_diagnosis_tai_color/save_weights_tai_color/resNet_50.ckpt'
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
        model.load_weights(weights_path)
        # prediction
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)

        print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        self.tai_color.setText(" {} ".format(class_indict[str(predict_class)]))
    def tongue_shape_predict(self):
        im_height = 224
        im_width = 224
        num_classes = 3
        img_path = "./img/jieguo.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        #    resize image to 224x224
        img = img.resize((im_width, im_height))
        # scaling pixel value to (0-1)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))
        # read class_indict
        json_path = './tongue_diagnosis_design/tongue_diagosis_shape/class_indices_fat_thin.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        feature = resnet50(num_classes=3, include_top=False)
        feature.trainable = False
        model = tf.keras.Sequential([feature,
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(1024, activation="relu"),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(num_classes),
                                     tf.keras.layers.Softmax()])

        weights_path = './tongue_diagnosis_design/tongue_diagosis_shape/save_weights_fat_thin/resNet_50.ckpt'
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
        model.load_weights(weights_path)
        # prediction
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)

        print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        self.tongue_shape.setText("{}".format(class_indict[str(predict_class)]))
    def tongur_shape_crack_predict(self):
        im_height = 224
        im_width = 224
        num_classes = 2
        img_path = "./img/jieguo.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        #    resize image to 224x224
        img = img.resize((im_width, im_height))
        # scaling pixel value to (0-1)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))
        # read class_indict
        json_path = './tongue_diagnosis_design/tongue_diagosis_shape/class_indices_crack.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        feature = resnet50(num_classes=2, include_top=False)
        feature.trainable = False
        model = tf.keras.Sequential([feature,
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(1024, activation="relu"),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(num_classes),
                                     tf.keras.layers.Softmax()])

        weights_path = './tongue_diagnosis_design/tongue_diagosis_shape/save_weights_crack/resNet_50.ckpt'
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
        model.load_weights(weights_path)
        # prediction
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)

        print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        self.tongue_shape2.setText("{}".format(class_indict[str(predict_class)]))
    def tongue_shape_intented_predict(self):
        im_height = 224
        im_width = 224
        num_classes = 2
        img_path = "./img/jieguo.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        #    resize image to 224x224
        img = img.resize((im_width, im_height))
        # scaling pixel value to (0-1)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))

        # read class_indict
        json_path = './tongue_diagnosis_design/tongue_diagosis_shape/class_indices_intented.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        feature = resnet50(num_classes=2, include_top=False)
        feature.trainable = False
        model = tf.keras.Sequential([feature,
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(1024, activation="relu"),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(num_classes),
                                     tf.keras.layers.Softmax()])

        weights_path = './tongue_diagnosis_design/tongue_diagosis_shape/save_weights_intented/resNet_50.ckpt'
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
        model.load_weights(weights_path)
        # prediction
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)

        print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        self.tongue_shape3.setText(" {} ".format(class_indict[str(predict_class)]))
    def esophagus_predict(self):
        im_height = 224
        im_width = 224
        num_classes = 2
        # load image
        img_path = "./img/jieguo.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        #    resize image to 224x224
        img = img.resize((im_width, im_height))
        # scaling pixel value to (0-1)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))
        # read class_indict
        json_path = './esophagus_cancer_classification/class_indices_esophagus.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        json_file = open(json_path, "r")
        class_indict = json.load(json_file)
        feature = resnet50(num_classes=2, include_top=False)
        feature.trainable = False
        model = tf.keras.Sequential([feature,
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(1024, activation="relu"),
                                     tf.keras.layers.Dropout(rate=0.5),
                                     tf.keras.layers.Dense(num_classes),
                                     tf.keras.layers.Softmax()])

        weights_path = './esophagus_cancer_classification/sample_weights/resNet_50.ckpt'
        assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
        model.load_weights(weights_path)
        # prediction
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        gl = "{:.3}".format(result[predict_class])
        xx = "{}".format(class_indict[str(predict_class)])
        if xx.strip() == "esophagus_cancer":
            self.esophagus_sug.setText("如果您在吞咽粗硬食物时感到不同程度的不适感觉\n包括咽下食物哽噎感、胸骨后烧灼样、针刺样或牵拉摩擦样疼痛时")
            self.esophagus_sug.append("那么您将有"+str(gl)+"的概率是"+str(xx))
            self.esophagus_sug.append("\n")
            self.esophagus_sug.append("请及时就医！\n")
            self.esophagus_sug.append("如若没有上述明显症状，可以忽略此诊断建议")
        elif xx.strip() == "non_esophagus_cancer":
            self.esophagus_sug.setText("您没有食管癌舌象的特征！")
    def suggestion_function(self):

        a = self.tongue_color.text()  #舌色
        b = self.tai_color.text()     #苔色
        c = self.tongue_shape.text()   #胖大、瘦小
        d = self.tongue_shape2.text()  #裂纹
        e = self.tongue_shape3.text()  #齿痕



        if a.strip() == 'dark_red_tongue':
           self.sugtext.setText("绛红舌：主热入营血，耗伤营阴\n")
        elif a.strip() == 'light_white_tongue':
           self.sugtext.setText("淡白舌：主气血两虚、阳虚。常见于贫血，重度营养不良，慢性消化系统、呼吸系统、心血管系统的疾病\n")
        elif a.strip() == 'purple_tongue':
           self.sugtext.setText("青紫舌：主气血运行不畅，血瘀。一般青紫程度与淤血的程度相关，淤血越重，青紫紫暗的程度越重。\n")
        elif a.strip() == 'red_tongue':
           self.sugtext.setText("红舌：主热证，一般舌质愈红，提示热势愈甚\n")
        elif a.strip() == 'reddish_tongue_backups':
           self.sugtext.setText("淡红舌：心气充足，胃气旺盛，气血调和，常见于正常人或病情轻浅阶段。\n")

        if b.strip() == 'white_coating':
            self.sugtext.append("白苔：可见于正常人，也主表证、寒证。\n")
        elif b.strip() == 'yellow_coating':
            self.sugtext.append("黄苔：主里证，热证\n")
        elif b.strip() == 'gray_black_coating':
            self.sugtext.append("灰黑苔：主里证，主热证又主寒证\n")


        if c.strip() == 'fat_tongue':
            self.sugtext.append("胖大舌：多主水湿内停，肿胀舌主心脾热盛、外感湿热\n")
        elif c.strip() == 'thin_tongue':
            self.sugtext.append("瘦薄舌主气血不足、阴虚火旺，舌色红或者红绛且瘦小的，为气阴\n")
        elif c.strip() == 'normal_tongue':
            self.sugtext.append("\n")
        if d.strip() == 'crack_tongue':
            self.sugtext.append("裂纹舌：提示精血亏虚或阴津耗损，是舌体失养，甚至全身营养不良的一种表现。舌色浅淡而裂纹者，是血虚之候；舌色红绛而有裂纹，则由热盛伤津，阴津耗损所致。\n")
        if d.strip() == 'normal_tongue':
            self.sugtext.append("\n")


        if e.strip() == 'intented_tongue':
            self.sugtext.append("主脾虚、水湿内盛证，舌胖大而多齿痕多属脾虚或湿困\n")
        if e.strip() == 'normal_tongue':
            self.sugtext.append("\n")
    def back_home(self):
        self.segoutput.setPixmap(QPixmap('back2.jpg'))
        self.uesr_tongue.setPixmap(QPixmap('back2.jpg'))
        self.tongue_recognition.setPixmap(QPixmap('back2.jpg'))
        self.tongue_color.setText("")
        self.tai_color.setText("")
        self.tongue_shape.setText("")
        self.tongue_shape2.setText("")
        self.tongue_shape3.setText("")
        self.sugtext.setText("")
        self.esophagus_sug.setText("")
        second_main.close()
        main_win.show()







if __name__ == '__main__':
    # 下面是使用PyQt5的固定用法
    app = QApplication(sys.argv)
    #初始化窗口
    main_win = mainWin()
    second_main = secondmain()
    #second_main.setFixedSize(1666,920)
    main_win.setFixedSize(1666,870)
    #main_win.move((QApplication.desktop().width()-main_win.width())/2,(QApplication.desktop().height()-main_win.height())/13)

    main_win.show()
    sys.exit(app.exec_())