#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:36:15 2025

@author: slash
"""
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import tf2onnx
import onnx


tf.keras.backend.set_image_data_format('channels_first')
#tf.keras.backend.set_image_data_format('channels_last')


# 搭建LeNet网络
net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6,kernel_size=5,activation='sigmoid',input_shape=(1, 28,28)),
#    tf.keras.layers.Conv2D(filters=24,kernel_size=5,activation='sigmoid',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Conv2D(filters=48,kernel_size=5,activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Flatten(),

#    accuracy: 0.99
    tf.keras.layers.Dense(120,activation='relu'),
    tf.keras.layers.Dense(84,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')

#    accuracy: 0.8
#    tf.keras.layers.Dense(120,activation='sigmoid'),
#    tf.keras.layers.Dense(84,activation='sigmoid'),
#    tf.keras.layers.Dense(10,activation='sigmoid')

])

#fashion_mnist = tf.keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

train_images = tf.reshape(train_images, (train_images.shape[0], 1, train_images.shape[1], train_images.shape[2]))
# #print(train_images.shape[2])
# #print(train_images.shape)
test_images = tf.reshape(test_images,(test_images.shape[0],1, test_images.shape[1],test_images.shape[2]))

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.0, nesterov = False)
net.compile(optimizer = optimizer,
           loss = 'sparse_categorical_crossentropy',
           metrics = ['accuracy'])


net.summary()

net.fit(train_images, train_labels, epochs = 50, validation_split = 0.1)

test_loss, test_acc = net.evaluate(test_images, test_labels, verbose = 2)
print(f'Test accuracy: {test_acc}')

tf.saved_model.save(net, "./saved_model_letnet")
spec = (tf.TensorSpec((1, 1, 28, 28), tf.float32, name = "input"),)
onnx_model, _ = tf2onnx.convert.from_keras(net, input_signature = spec,
                                            opset = 15,
                                            output_path = "./saved_model_letnet/letnet_model.onnx")
