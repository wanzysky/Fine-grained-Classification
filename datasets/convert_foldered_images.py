from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob

import numpy as np
import cv2
import random
import tensorflow as tf

from datasets import dataset_utils

INPUT_DIRS = [
        '/data/JD_dataset/trainfull_v2/',
        '/data/JD_dataset/trainfull_v2/rotate',
        '/data/JD_dataset/trainfull_v2/gaussian'
        '/data/JD_dataset/trainfull_v2/brightness',
        '/data/JD_dataset/trainfull_v2/saturation']
OUTPUT_DIR = '/data/JD_dataset/augmentation'

NUM_SHARDS = 25

_IMAGE_HEIGHT = int(720 / 2)
_IMAGE_WIDTH = int(1280 / 2)


def _get_label(image_path):
    basename = os.path.split(image_path)[0]
    return int(basename.split('/')[-1]) - 1


def _encode_image(sess, image):
    pass

def run():
    if not tf.gfile.Exists(OUTPUT_DIR):
        tf.gfile.MakeDirs(OUTPUT_DIR)

    input_queue = []
    for dirname in INPUT_DIRS:
        input_queue += glob.glob(os.path.join(dirname, '*/*.jpg'))
    random.shuffle(input_queue)

    output_path = os.path.join(OUTPUT_DIR, 'jdpig_test_%05d-of-%05d.tfrecord')
    records_per_shard = len(input_queue) / (NUM_SHARDS - 1)
    print('%d images to convert, %d per shard' % (len(input_queue), records_per_shard))
    all_count = 0
    with tf.Graph().as_default():
        with tf.Session('') as sess:
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded = tf.image.encode_jpeg(image_placeholder)
            count = 0
            record_writer = None
            shard_id = 0
            record_writer = tf.python_io.TFRecordWriter(output_path % (shard_id, NUM_SHARDS - 1))
            for image_path in input_queue:
                count += 1
                all_count += 1
                image = cv2.imread(image_path)
                if image is not None:
                    if count % 100 == 0:
                        print('writing to %d/%d in shard %d' % (count, records_per_shard, shard_id))
                    print('reading %dth image in shard %s' % (all_count, shard_id))
                    label = _get_label(image_path)
                    encoded_img = sess.run(encoded, feed_dict={image_placeholder: image})
                    example = dataset_utils.image_to_tfexample(encoded_img, b'jpg', _IMAGE_HEIGHT, _IMAGE_WIDTH, label)
                    record_writer.write(example.SerializeToString())

                    if records_per_shard <= count:
                        shard_id += 1
                        count = 0
                        record_writer = tf.python_io.TFRecordWriter(output_path % (shard_id, NUM_SHARDS))
