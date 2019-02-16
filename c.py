import tensorflow as tf
import cv2

image_list = []
label_list = []

for i in range(1,855):
    name = './suhyeong/'+str(i) + '.jpg'
    image_list.append(name)
    label_list.append(1)


options = tf.python_io.TFRecordOptions(compression_type = tf.python_io.TFRecordCompressionType.GZIP)
writer = tf.python_io.TFRecordWriter('a.tfrecords')
for image_name,label in zip(image_list,label_list):

    image = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)

    # print(type(image))


    _image = image.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [_image])),
        'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
    }))

    writer.write(example.SerializeToString())

writer.close()
