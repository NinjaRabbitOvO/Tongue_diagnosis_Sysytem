import os
import glob
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from unet_keras.tongue_recognition.tongue_recognition_model import resnet50


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./."))  # get FundusVessels root path
    image_path = os.path.join(data_root, "datasets", "train_tongue_recognition")  # flower FundusVessels set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    im_height = 224
    im_width = 224
    batch_size = 16
    epochs = 25
    num_classes = 2

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    def pre_function(img):
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        return img
    #在利用图像数据进行深度学习建模的任务中，如果数据集较小，
    # 我们需要进行Image Data Augmentation：对已有图片进行平移，剪切，垂直对称等操作形成新的图片。
    # 将新图片加入数据集，从而扩充数据集。Keras的内置函数ImageDataGenerator就是用来扩充图像数据集的。
    train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                               preprocessing_function=pre_function)

    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)
    #以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict分类字典
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices_tongue_recognition.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    # img, _ = next(train_data_gen)
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    feature = resnet50(num_classes=2, include_top=False)

    pre_weights_path = './tf_resnet50_weights/pretrain_weights.ckpt'
    assert len(glob.glob(pre_weights_path+"*")), "cannot find {}".format(pre_weights_path)
    feature.load_weights(pre_weights_path)
    feature.trainable = False
    feature.summary()

    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])
    # model.build((None, 224, 224, 3))
    model.summary()

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def val_step(images, labels):
        output = model(images, training=False)
        loss = loss_object(labels, output)

        val_loss(loss)
        val_accuracy(labels, output)

    best_test_loss = float('inf')
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        #train
        train_bar = tqdm(range(total_train // batch_size))
        for step in train_bar:
            images, labels = next(train_data_gen)
            train_step(images, labels)

            # print 训练集 process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # validate
        val_bar = tqdm(range(total_val // batch_size), colour='green')
        for step in val_bar:
            test_images, test_labels = next(val_data_gen)
            val_step(test_images, test_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())

        if val_loss.result() < best_test_loss:
            best_test_loss = val_loss.result()
            model.save_weights("./save_weights_tongue_recognition/resNet_50.ckpt", save_format="tf")


if __name__ == '__main__':
    main()