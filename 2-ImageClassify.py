import tensorflow as tf
from tensorflow import keras
#Keras是一个由Python编写的开源人工神经网络库，进行深度学习模型的设计、调试、评估、应用和可视化
import numpy as np
import matplotlib.pyplot as plt
#print(tf.__version__)# 显示当前TensorFlow的版本
#导入数据集
fashion_mnist = keras.datasets.fashion_mnist
#keras内部集成了这个数据集，先将其抓取下来
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#分成不同的数组管理起来
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#服装有哪些种类，与0-9的编号一一对应
# print(train_images.shape)
# print(len(train_labels))
#处理数据，这一步其实很关键，这样就知道这个数据是什么格式，如何统一处理，因为现在这个数据集已经处理成标准形式了
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# train_images = train_images / 255.0#让数据落在0-1范围内
# test_images = test_images / 255.0
# plt.figure(figsize=(10, 10))#显示的格式
# for i in range(25):
#     plt.subplot(5, 5, i+1)#显示的格式
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
#建立神经网络层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #reformat the data
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(10)
    ])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
