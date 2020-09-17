#引入包
#Keras是一个由Python编写的开源人工神经网络库，可以作为Tensorflow、Microsoft-CNTK和Theano的高阶应用程序接口，进行深度学习模型的设计、调试、评估、应用和可视化
import tensorflow as tf
from tensorflow import keras
#数据处理
testset = [1,2,3]


#神经网络设计
number1 = 1
number2 = 2
number3 = 3
model = tf.keras.Sequential()
model.add(keras.layers.Dense(number1, activation=''))
model.add(keras.layers.Dense(number2, activation=''))
model.add(keras.layers.Dense(number3, activation=''))

#神经网络训练及可视化展示
model.compile(optimizer=' ', loss=' ', metrics=[' '])
model.fit(testset, epochs=10)

#其他（如何导出，如何使用等等）