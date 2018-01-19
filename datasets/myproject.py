"""
Provides data for the flowers dataset.
The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s_*.tfrecord'


#####################################################################################
###########增加与此函数相关代码以自动获取到_NUM_VALIDATION、_NUM_TRAIN、_NUM_CLASSES等参数
_DATA_DIR= './MyProject'
_DATA_FOLDER_NAME = 'photos_to_be_converted'
_VALIDATION_PERCENTAGE = 0.1


def _get_filenames_and_classes(dataset_dir):
  data_root = os.path.join(dataset_dir, _DATA_FOLDER_NAME)
  directories = []
  class_names = []
  for filename in os.listdir(data_root):
    path = os.path.join(data_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


_NUM_VALIDATION = round(len(_get_filenames_and_classes(_DATA_DIR)[0])*_VALIDATION_PERCENTAGE)

_NUM_TRAIN = len(_get_filenames_and_classes(_DATA_DIR)[0]) - _NUM_VALIDATION

_NUM_CLASSES = len(_get_filenames_and_classes(_DATA_DIR)[1]) 

SPLITS_TO_SIZES = {'train': _NUM_TRAIN, 'validation': _NUM_VALIDATION}

#####################################################################################
#####################################################################################




_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}



def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)

