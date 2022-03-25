#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import colorsys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from unet_keras.yolov4_pytorch_master.nets.yolo4 import YoloBody
from unet_keras.yolov4_pytorch_master.utils.utils import (DecodeBox, letterbox_image,
                                                          non_max_suppression, yolo_correct_boxes)


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : './yolov4_pytorch_master/logs/Epoch71-Total_Loss1.8202-Val_Loss2.5562.pth',
        "anchors_path"      : './yolov4_pytorch_master/model_data/yolo_anchors.txt',
        "classes_path"      : './yolov4_pytorch_master/model_data/voc_classes.txt',
        #单独运行yolo代码时、调用video时
        #"model_path": './logs/Epoch71-Total_Loss1.8202-Val_Loss2.5562.pth',
        #"anchors_path": './model_data/yolo_anchors.txt',
        #"classes_path": './model_data/voc_classes.txt',
        "model_image_size"  : (416, 416, 3),
        "confidence"        : 0.5,
        "iou"               : 0.3,
        "cuda"              : False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolov4模型
        #---------------------------------------------------#
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()
        #我们在这里会得到yolohead的三个特征层的输出
        #---------------------------------------------------#
        #   载入yolov4模型的权重
        #---------------------------------------------------#
        print('Loading weights into state dict...')
        #是否使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #下面两行代码是torch提供的加载模型以及预训练权重的函数
        state_dict = torch.load(self.model_path, map_location=device)

        self.net.load_state_dict(state_dict)

        print('YOLOv4模型和权重加载完毕!')
        
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        #---------------------------------------------------#
        #   建立三个特征层的解码
        #---------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))
        #当i=0时，self.yolo_decodes.append(DecodeBox([[288 193] [310 219] [336 304]]，1，416,416)
        #先看一下anchors里面有什么：
        """
        [[[288. 193.]
          [310. 219.]
         [336. 304.]]

        [[189. 244.]
         [239. 149.]
        [249.
        293.]]

        [[126.  84.]
         [166. 199.]
        [172.
        132.]]]
        """


        #print("对yolohead的输出进行解码的结果")  #里面存放的是三个特征层的解码结果
        #print(self.yolo_decodes)
        #print("yolohead解码结果的维度")
        #print(self.yolo_decodes.size())

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():  #不能进行梯度计算的上下文管理器。当你确定你不调用Tensor.backward()时，不能计算梯度对测试来讲非常有用。
            # 对计算它将减少内存消耗
            images = torch.from_numpy(np.asarray(images))    #从numpy.ndarray创建一个张量。
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images) #制一部得到的是yolohead的输出结果
            #yolohead一共得到三个张量，即三个特征层，分别为（N,18,13,13) (1,18,26,26) (1,18,52,52)
            #print("yolohead的输出结果：")
            #for i in range(3):
            #    print(outputs[i].size())

            #print(outputs)

            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            # 在这一步将会得到三个特征层解码的输出结果
            # 输出结果的形式为#(1,3,52,52,6)  (1,3,26,26,6)   (1,3,13,13,6) 也就是此时每个特征层上都有三个预测框
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            #print("三个特征层解码结果的堆叠：")  # 也就是将这九个框堆叠到一张图像上
            #print(output)
            #print(output.size())  # （1,10647，6） 10647=（52*52+26*26+13*13）*3
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)

            #print("进行非极大抑制处理之后：")
            # print(batch_detections)
            #print(batch_detections)  # [tensor([[130.3967, 114.1140, 245.2802, 211.4902,   0.9733,   1.0000,   0.0000]])]

            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            # ---------------------------------------------------------#
            #   将进行过非极大抑制之后的输出结果分开，以便进行下一步的调整预测框
            # ---------------------------------------------------------#
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            #print("进行非极大抑制之后的top_index")
            #print(top_index)
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            #print("进行非极大抑制之后的top_conf")
            #print(top_conf)
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            #print("进行非极大抑制之后的top_label")
            #print(top_label)

            top_bboxes = np.array(batch_detections[top_index, :4])
            #print("进行非极大抑制之后的top_bboxes")
            #print(top_bboxes)
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
            #print("去掉之前加上灰条之后的boxes")
            #print(boxes)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]


            #进行微调  这一步是借鉴的别人的经验
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            #print(label, top, left, bottom, right)


            # 参数说明
            # 第一个参数 开始截图的x坐标
            # 第二个参数 开始截图的y坐标
            # 第三个参数 结束截图的x坐标
            # 第四个参数 结束截图的y坐标
            bbox = (left,top, right,bottom)   #四个点的坐标
            im = image.crop(bbox)

            # 参数 保存截图文件的路径
            im.save('./tonguerecbtn_image/user_tonguerec.jpg')
            #im.save('../tonguerecbtn_image/user_tonguerec.jpg')
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return im

    def detect_image_origi(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            #在这一步将会得到三个特征层解码的输出结果
            #输出结果的形式为#(1,3,52,52,6)  (1,3,26,26,6)   (1,3,13,13,6) 也就是此时每个特征层上都有三个预测框
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            #print("三个特征层解码结果的堆叠：") #也就是将这九个框堆叠到一张图像上
            #print(output)
            #print(output.size())   #（1,10647，6） 10647=（52*52+26*26+13*13）*3
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            #print("进行非极大抑制处理之后：")
            #print(batch_detections)
            #print(batch_detections)#[tensor([[130.3967, 114.1140, 245.2802, 211.4902,   0.9733,   1.0000,   0.0000]])]

            # ---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            # ---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            # ---------------------------------------------------------#
            #   将进行过非极大抑制之后的输出结果分开，以便进行下一步的调整预测框
            # ---------------------------------------------------------#
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            #print("进行非极大抑制之后的top_index")
            #print(top_index)
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            #print("进行非极大抑制之后的top_conf")
            #print(top_conf)
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            #print("进行非极大抑制之后的top_label")
            #print(top_label)

            top_bboxes = np.array(batch_detections[top_index, :4])
            #print("进行非极大抑制之后的top_bboxes")
            #print(top_bboxes)
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            # -----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            # -----------------------------------------------------------------#
            boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                       np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
            #print("去掉之前加上灰条之后的boxes")
            #print(boxes)
            #这个时候获取到的就是我们最后的预测框，所以在整个的预测过程中，我们用到的基于真实标签框产生的那九个先验框起到的是调整为对应特征层大小的框的作用
            #直白点来讲，就是13*13的特征层我们用这种坐标来表示这些先验框



        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            #print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
