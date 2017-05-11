# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow_models.show_tell_attend.ops.image_embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from show_tell_attend.ops import image_embedding


class VGG19Test(tf.test.TestCase):

  def setUp(self):
    super(VGG19Test, self).setUp()

    batch_size = 4
    height = 224
    width = 224
    num_channels = 3
    self._images = tf.placeholder(tf.float32,
                                  [batch_size, height, width, num_channels])
    self._batch_size = batch_size

  def _countVGGParameters(self):
    """Counts the number of parameters in the vgg model at top scope."""
    counter = {}
    fcn = {"fc6","fc7","fc8"}
    for v in tf.global_variables():
      name_tokens = v.op.name.split("/")
      if name_tokens[0]=="vgg_19" and name_tokens[1] not in fcn:
        name = "vgg_19/" + name_tokens[1] + "/" + name_tokens[2]
        num_params = v.get_shape().num_elements()
        assert num_params
        counter[name] = counter.get(name, 0) + num_params
      elif name_tokens[0]=="vgg_19" and name_tokens[1] in fcn:
        name = "vgg_19/" + name_tokens[1]
        num_params = v.get_shape().num_elements()
        assert num_params
        counter[name] = counter.get(name, 0) + num_params
    return counter

  def _verifyParameterCounts(self):
    """Verifies the number of parameters in the vgg model."""
    param_counts = self._countVGGParameters()
    expected_param_counts = {
        "vgg_19/conv1/conv1_1": 1920,
        "vgg_19/conv1/conv1_2": 37056,
        "vgg_19/conv2/conv2_1": 74112,
        "vgg_19/conv2/conv2_2": 147840,
        "vgg_19/conv3/conv3_1": 295680,
        "vgg_19/conv3/conv3_2": 590592,
        "vgg_19/conv3/conv3_3": 590592,
        "vgg_19/conv3/conv3_4": 590592,
        "vgg_19/conv4/conv4_1": 1181184,
        "vgg_19/conv4/conv4_2": 2360832,
        "vgg_19/conv4/conv4_3": 2360832,
        "vgg_19/conv4/conv4_4": 2360832,
        "vgg_19/conv5/conv5_1": 2360832,
        "vgg_19/conv5/conv5_2": 2360832,
        "vgg_19/conv5/conv5_3": 2360832,
        "vgg_19/conv5/conv5_4": 2360832,
        "vgg_19/fc6": 102772736,
        "vgg_19/fc7": 16789504,
        "vgg_19/fc8": 4097000
    }
    self.assertDictEqual(expected_param_counts, param_counts)

  def _assertCollectionSize(self, expected_size, collection):
    actual_size = len(tf.get_collection(collection))

    if expected_size != actual_size:
      self.fail("Found %d items in collection %s (expected %d)." %
                (actual_size, collection, expected_size))

  def testTrainableTrueIsTrainingTrue(self):
    embeddings = image_embedding.vgg_19_extract(
        self._images, trainable=True, is_training=True)
    self.assertEqual([self._batch_size, 14, 14, 512], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(74, tf.GraphKeys.GLOBAL_VARIABLES)
    self._assertCollectionSize(38, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(36, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(19, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(43, tf.GraphKeys.SUMMARIES)

  def testTrainableTrueIsTrainingFalse(self):
    embeddings = image_embedding.vgg_19_extract(
        self._images, trainable=True, is_training=False)
    self.assertEqual([self._batch_size, 14, 14, 512], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(74, tf.GraphKeys.GLOBAL_VARIABLES)
    self._assertCollectionSize(38, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(19, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(43, tf.GraphKeys.SUMMARIES)

  def testTrainableFalseIsTrainingTrue(self):
    embeddings = image_embedding.vgg_19_extract(
        self._images, trainable=False, is_training=True)
    self.assertEqual([self._batch_size, 14, 14, 512], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(74, tf.GraphKeys.GLOBAL_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(43, tf.GraphKeys.SUMMARIES)

  def testTrainableFalseIsTrainingFalse(self):
    embeddings = image_embedding.vgg_19_extract(
        self._images, trainable=False, is_training=False)
    self.assertEqual([self._batch_size, 14, 14, 512], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(74, tf.GraphKeys.GLOBAL_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(43, tf.GraphKeys.SUMMARIES)


if __name__ == "__main__":
  tf.test.main()
