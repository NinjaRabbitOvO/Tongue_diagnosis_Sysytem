# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet_keras.yolov4_pytorch_master.nets.yolo4 import YoloBody
from unet_keras.yolov4_pytorch_master.nets.yolo_training import Generator, YOLOLoss
from unet_keras.yolov4_pytorch_master.utils.dataloader import YoloDataset, yolo_dataset_collate


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    # print(np.array(anchors)) [ 12.  16.  19.  36.  40.  28.  36.  75.  76.  55.  72. 146. 142. 110.192. 243. 459. 401.]
    # print(np.array(anchors).reshape([-1,3,2])[::-1,:,:])
    """
    [[[142. 110.]
    [192. 243.]
     [459. 401.]]

     [[ 36.  75.]
     [ 76.  55.]
     [ 72. 146.]]

    [[ 12.  16.]
     [ 19.  36.]
     [ 40.  28.]]]
     """
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = net(images)
            # 这里的outputs是yolohead的原始输出（1,13,13,18）（1,26,26,18）（1,52,52,18）
            losses = []
            num_pos_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for i in range(3):  # 三个特征层
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos
                print("第" + str(i) + "个特征层的losses")
                print(losses[i])
                print(num_pos)
            loss = sum(losses) / num_pos_all
            print("三个特征层的总loss")
            print(sum(losses))
            print("如果我的想法是正确的，那么此时的total_loss应该为：")
            print(loss)
            # print(loss)

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                # print(num_pos_all)
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = False
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = False
    # -------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    # -------------------------------#
    input_shape = (416, 416)

    # ----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # ----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'
    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    """
       [[[142. 110.]
       [192. 243.]
        [459. 401.]]

        [[ 36.  75.]
        [ 76.  55.]
        [ 72. 146.]]

       [[ 12.  16.]
        [ 19.  36.]
        [ 40.  28.]]]
        """
    num_classes = len(class_names)  # 一类

    # ------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False 
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    # ------------------------------------------------------#
    mosaic = False
    Cosine_lr = False
    smoooth_label = 0

    # ------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改classes_path和对应的txt文件
    # ------------------------------------------------------#
    # print(len(anchors[0]))  三个先验框
    model = YoloBody(len(anchors[0]), num_classes)
    # print("model:"+str(model))

    # model_path = "logs/Epoch71-Total_Loss1.8202-Val_Loss2.5562.pth"
    model_path = "model_data/yolo4_voc_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()  # 状态字典：保存模型中的weight权值和bias偏置值
    # 在PyTorch中，state_dict是一个从参数名称映射到参数Tesnor的字典对象。
    pretrained_dict = torch.load(model_path, map_location=device)  # 加载模型的权重文件
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()
    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # model.train()是保证BN层能够用到每一批数据的均值和方差。
    # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。

    if Cuda:
        net = torch.nn.DataParallel(model)
        # 在Pytorch中，nn.DataParallel函数可以调用多个GPU，帮助加速训练。
        # 在这里CUda没有用
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):  # 对于每一个特征层都计算其loss值？
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, \
                                    (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize))
    print("三个特征层loss值的汇总：")
    print(yolo_losses)
    # 三个特征层loss值的汇总：[YOLOLoss(), YOLOLoss(), YOLOLoss()]
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    annotation_path = '2007_train.txt'
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-3
        Batch_size = 4
        Init_Epoch = 0
        Freeze_Epoch = 50

        # ----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        # ----------------------------------------------------------------------------#
        optimizer = optim.Adam(net.parameters(), lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic,
                                        is_train=True)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(train=True, mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(train=False, mosaic=mosaic)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 2
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic,
                                        is_train=True)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(train=True, mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(train=False, mosaic=mosaic)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()
