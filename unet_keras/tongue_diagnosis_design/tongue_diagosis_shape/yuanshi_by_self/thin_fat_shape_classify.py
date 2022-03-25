#构建数据集
import os
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import pylab
import tensorflow as tf

#这个函数的作用就是 在指定的目录下读取我们的train数据
#读取指定目录下的图片文件信息，返回文件名列表和标签列表
from keras.preprocessing import image

def read_image_filenames(data_dir):
    thin_tongue_dir = data_dir + 'thin_tongue/'
    normal_tongue_dir = data_dir + 'normal_tongue/'
    fat_tongue_dir = data_dir + 'fat_tongue/'
    #构建特征数据集，值为对应的图片文件名
    thin_tongue_filenames = tf.constant([thin_tongue_dir + fn for fn in os.listdir(thin_tongue_dir)])
    #瘦小舌 0-116    0
    normal_tongue_filenames = tf.constant([normal_tongue_dir + fn for fn in os.listdir(normal_tongue_dir)])
    #正常舌 0-76    1
    fat_tongue_filenames = tf.constant([fat_tongue_dir + fn for fn in os.listdir(fat_tongue_dir)])
    #胖大舌0-58 2
    filenames = tf.concat([thin_tongue_filenames,normal_tongue_filenames,fat_tongue_filenames],axis=-1)

    labels = tf.concat([
        tf.zeros(thin_tongue_filenames.shape,dtype=tf.int32),
        tf.ones(normal_tongue_filenames.shape,dtype=tf.int32),
        tf.fill(fat_tongue_filenames.shape, 2),
    ],
        axis=-1
    )
    return filenames,labels

#读取图片文件并解码，调整图片的大小并标准化
def decode_image_and_resize(filename,label):
    image_string = tf.io.read_file(filename)   #读取图片，这个时候只是读取出来成了字符串格式
    image_decoded = tf.image.decode_jpeg(image_string)  #解码为jpeg
    #调整图片大小
    image_resized = tf.image.resize(image_decoded,[224,224]) / 255.0
    return image_resized,label

#将上述步骤打包成一个函数，方便后期调用
def prepare_dataset(data_dir,buffer_size=283,batch_size=8):
    filenames, labels = read_image_filenames(train_data_dir)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # 图像预处理
    dataset = dataset.map(
        map_func=decode_image_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 下面进行乱序：
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    # 上面这一句代码实现了将原本的数据集分成若干个dataset_batch，每一个dataset_batch里面都有8个数据
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)#并行处理

    return dataset


def vgg16_model(input_shape=(224,224,3)):
    vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                              weights='imagenet',
                                              input_shape=input_shape)
    for layer in vgg16.layers:
        layer.trainable = False

    last = vgg16.output

    x = tf.keras.layers.Flatten()(last)
    x = tf.keras.layers.Dense(256,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(3,activation='softmax')(x)

    model = tf.keras.models.Model(inputs=vgg16.input, outputs=x)

    model.summary()
    return model


model = vgg16_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""


#恢复模型结构
with open('./model/tongue_shape_thin_fat.yaml') as yamlfile:
    loaded_model_yaml = yamlfile.read()
model = tf.keras.models.model_from_yaml(loaded_model_yaml)

model.load_weights('./model/tongue_shape_thin_fat.h5')
#正式开始读取数据集

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


"""


train_data_dir = '../train_shape/'
buffer_size = 200
batch_size = 8
dataset_train = prepare_dataset(train_data_dir,buffer_size,batch_size)


#定义超参数
training_epochs = 20
train_history = model.fit(dataset_train,epochs=training_epochs,verbose=1)

"""
#模型结构存储在.yaml文件中
yaml_string = model.to_yaml()
with open('./model/tongue_shape_thin_fat.yaml','w') as model_file:
    model_file.write(yaml_string)

model.save_weights('./model/tongue_shape_thin_fat.h5')

"""





def read_image_files(path,start,finish,image_size=(224,224)):
    test_files = os.listdir(path)
    test_images = []
    #读取测试图片并处理预处理
    for fn in test_files[start:finish]:
        img_filename = path + fn
        img = image.load_img(img_filename,target_size=image_size)
        img_array = image.img_to_array(img)
        test_images.append(img_array)
    test_data = np.array(test_images)
    test_data /= 255.0

    print("You choose the image %d to image %d" %(start,finish))
    print("The test_data's shape is",end=' ')
    print(test_data.shape)

    return test_data

def test_image_predict(path,start,finish,image_size=(224,224)):
    test_data = read_image_files(path,start,finish,image_size)

    preds = model.predict(test_data)

    for i in range(0,finish-start):
        if np.argmax(preds[i]) == 0:
            label = "thin_tongue " + str(preds[i][0])
        elif np.argmax(preds[i]) == 1:
            label = "normal_tongue "+str(preds[i][1])
        elif np.argmax(preds[i]) == 2:
            label = "fat_tongue " + str(preds[i][2])
        plt.title(label)
        plt.imshow(test_data[i])
        plt.axis('off')
        plt.show()


#从test目录下读取训练数据
test_data_dir = '../user_test/'
test_image_predict(test_data_dir,0,7)
