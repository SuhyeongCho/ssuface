import tensorflow as tf
import cv2
from glob import glob


def _parse_function(example_proto):
  features = {'image': tf.FixedLenFeature([], tf.string, default_value=""),
              'label': tf.FixedLenFeature([], tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features=features)
  image = tf.decode_raw(parsed_features['image'],tf.uint8)
  image = tf.reshape(image,[96,96])
  return image,parsed_features['label']

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
filenames = ["a.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

dataset = dataset.batch(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
# Compute for 100 epochs.
for _ in range(2):
    sess.run(iterator.initializer)
    while True:
        try:
            k,t = sess.run(next_element)
            cv2.imshow('a',k[0])
            cv2.waitKey(0)
        except tf.errors.OutOfRangeError:
            break

sess.close()








# reader = tf.TFRecordReader()
#
# tfrecords_filename = 'a.tfrecords'
# filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=10)
#
# _, serialized_example = reader.read(filename_queue)
#
# features={
#     'image': tf.FixedLenFeature([], tf.string),
#     'label': tf.FixedLenFeature([], tf.int64)
# }
#
# features = tf.parse_single_example(serialized_example,features=features)
#
# image = tf.decode_raw(features['image'],tf.uint8)
# image.set_shape([96*96])
# # image = tf.reshape(image,[96,96])
# label = tf.cast(features['label'],tf.int64)
# images_batch, labels_batch = tf.train.shuffle_batch([image,label],batch_size=20,capacity=854,min_after_dequeue=1)
#
#
#
# # Compute for 100 epochs.
# for _ in range(100):
#   sess.run(iterator.initializer)
#   while True:
#     try:
#       sess.run(next_element)
#     except tf.errors.OutOfRangeError:
#       break







# # read file
# dataset = tf.data.TFRecordDataset(filename)
# # parse each instance
# dataset = dataset.map(your_parser_fun, num_parallel_calls=num_threads)
# # preprocessing, e.g. scale to range [0, 1]
# dataset = dataset.map(some_preprocessing_fun)
# # shuffle
# dataset = dataset.shuffle(buffer_size)
# # form batch and epoch
# dataset = dataset.batch(batch_size)
# dataset = dataset.repeat(num_epoch)
# iterator = dataset.make_one_shot_iterator()
# # get a batch
# x_batch, y_batch = self.iterator.get_next()
#
# # do calculations
