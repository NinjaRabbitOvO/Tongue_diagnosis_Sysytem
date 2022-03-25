#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from unet_keras.unet import Unet
from PIL import Image
import os
unet = Unet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:         
        print('Open Error! Try again!')
        continue
    else:
        r_image = unet.detect_image(image)
        r_image.save('./img/jieguo.jpg')
        r_image.show()
"""
for i in range(53,123):
    # load image
    #img_path = "img/" + "normal"+ str(i) + ".jpg"
    img_path = "new_normal/" + str(i) +".jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    image = Image.open(img_path)
    r_image = unet.detect_image(image)
    print(str(i) +" 号处理好了！")
    r_image.save("new_normal2/"+ str(i)+".jpg")
"""