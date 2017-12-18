from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2

slim = tf.contrib.slim


data_pattern = '/Volumes/Wanzy/data/pig-tf-record/data_00004-of-00020.tfrecord'

keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 29',
}

items_to_handlers = {
  'image': slim.tfexample_decoder.Image(),
  'label': slim.tfexample_decoder.Tensor('image/class/label'),
}

decoder = slim.tfexample_decoder.TFExampleDecoder(
  keys_to_features, items_to_handlers)

labels_to_names = {}
for i in range(30):
    labels_to_names[i] = i
with tf.Session():
    data_set = slim.dataset.Dataset(
        data_sources=data_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=4658,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=30,
        labels_to_names=labels_to_names)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        data_set,
        num_readers=2,
        common_queue_capacity=20,
        common_queue_min=10
    )
    image, label = provider.get(['image', 'label'])
    with tf.Session('') as sess:
        with slim.queues.QueueRunners(sess):
            while True:
                im, l = sess.run([image, label])
                print(l)
                print(im.shape)
                cv2.imshow('%d' % int(l), im)
                cv2.waitKey()

