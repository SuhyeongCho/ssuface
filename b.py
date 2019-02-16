import tensorflow as tf
import numpy as np

size = 96
X = tf.placeholder(np.float32,shape=[None,size,size,1])

conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2])
conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2])
conv3 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[2,2],padding='SAME', activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2])

pool_flat = tf.reshape(pool3,[-1,size*size*2])

fc1 = tf.layers.dense(inputs=pool_flat,units=1024,activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1,units=256,activation=tf.nn.relu)
logits = tf.layers.dense(inputs=fc2,units=2)

print(X.get_shape())
print(conv1.get_shape())
print(pool1.get_shape())
print(conv2.get_shape())
print(pool2.get_shape())
print(conv3.get_shape())
print(pool3.get_shape())
print(pool_flat.get_shape())
print(fc1.get_shape())
print(fc2.get_shape())
print(logits.get_shape())
