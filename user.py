# 在combination中已经训练好了样本，在这里进行读入的操作
# 导入相应的库
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time

start = time.perf_counter()

# 设置图片的高和宽，一次训练所选取的样本数，迭代次数
im_height = 250
im_width = 250
batch_size = 256
epochs = 10

image_path = "./data/"  # 猫狗数据集路径
train_dir = "./data/train"  # 训练集路径
validation_dir = "./data/test1"  # 验证集路径

# 定义训练集图像生成器，并进行图像增强
# ImageDataGenerator通过实时数据产生一系列的旋转、平移、剪切、反转等变换过的图像
train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                           rotation_range=40,  # 旋转范围
                                           width_shift_range=0.2,  # 水平平移范围
                                           height_shift_range=0.2,  # 垂直平移范围
                                           shear_range=0.2,  # 剪切变换的程度
                                           zoom_range=0.2,  # 剪切变换的程度
                                           horizontal_flip=True,  # 水平翻转
                                           fill_mode='nearest')

# 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
# 为什么要用one-hot编码？因为猫狗之间是无序的，如果用1、2就会有大小的区别，
# 故使用01 10式的编码，使其只有一位有效
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,  # 从训练集路径读取图片
                                                           batch_size=batch_size,  # 一次训练所选取的样本数
                                                           shuffle=True,  # 打乱标签
                                                           target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                           class_mode='categorical')  # one-hot编码

# 训练集样本数
total_train = train_data_gen.n

# 定义验证集图像生成器，并对图像进行预处理
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # 归一化

# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,  # 从验证集路径读取图片
                                                              batch_size=batch_size,  # 一次训练所选取的样本数
                                                              shuffle=False,  # 不打乱标签
                                                              target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                              class_mode='categorical')  # one-hot编码

# 验证集样本数
total_val = val_data_gen.n

# 测试读取图片用时
end = time.perf_counter()
print('读取图片用时: %s Seconds' % (end - start))

start = time.perf_counter()
# Restore the weights
model = tf.keras.Sequential()
covn_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(250, 250, 3))
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
model.add(tf.keras.layers.Dense(512, activation='relu'))  # 添加全连接层
model.add(tf.keras.layers.Dropout(rate=0.5))  # 添加Dropout层，防止过拟合
model.add(tf.keras.layers.Dense(2, activation='softmax'))  # 添加输出层(2分类)

model.load_weights('./save_weights/myNASNetMobile.ckpt')

end = time.perf_counter()
print('加载用时: %s Seconds' % (end - start))

#  获取数据集的类别编码
class_indices = train_data_gen.class_indices
# 将编码和对应的类别存入字典
inverse_dict = dict((val, key) for key, val in class_indices.items())
# 加载测试图片
img = Image.open("./data/test/c.jpg")

print("已加载图片")
# 将图片resize到250x250大小
img = img.resize((250, 250))
# 归一化
img1 = np.array(img) / 255.
# 将图片增加一个维度，目的是匹配网络模型
img1 = (np.expand_dims(img1, 0))
# 将预测结果转化为概率值
result = np.squeeze(model.predict(img1))
predict_class = np.argmax(result)
print(inverse_dict[int(predict_class)], result[predict_class])
# 将预测的结果打印在图片上面
plt.title([inverse_dict[int(predict_class)], result[predict_class]])
# 显示图片
plt.imshow(img)
plt.show()
