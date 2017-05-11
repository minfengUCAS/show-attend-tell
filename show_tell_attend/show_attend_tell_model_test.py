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

"""Tests for tensorflow_models.im2txt.show_and_tell_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from show_tell_attend import configuration
from show_tell_attend import show_attend_tell_model


class ShowAttendTellModel(show_attend_tell_model.ShowAttendTellModel):
  """Subclass of ShowAndTellModel without the disk I/O."""

  def build_inputs(self):
    if self.mode == "inference":
      self.images = super(ShowAttendTellModel, self)._batch_norm(tf.random_uniform(
        shape=[self.config.batch_size, self.config.image_height,
               self.config.image_width, 3],
        minval=-1,
        maxval=1
      ))
      self.input_seqs = tf.random_uniform(
        [self.config.batch_size, 1],
        minval=0,
        maxval=self.config.vocab_size,
        dtype=tf.int64
      )
    else:
      # Replace disk I/O with random Tensors.
      self.images = super(ShowAttendTellModel, self)._batch_norm(tf.random_uniform(
          shape=[self.config.batch_size, self.config.image_height,
                 self.config.image_width, 3],
          minval=-1,
          maxval=1))
      self.input_seqs = tf.random_uniform(
          [self.config.batch_size, 32],
          minval=0,
          maxval=self.config.vocab_size,
          dtype=tf.int64)
      self.target_seqs = tf.random_uniform(
          [self.config.batch_size, 32],
          minval=0,
          maxval=self.config.vocab_size,
          dtype=tf.int64)
      self.input_mask = tf.ones_like(self.input_seqs)


class ShowAndTellModelTest(tf.test.TestCase):

  def setUp(self):
    super(ShowAndTellModelTest, self).setUp()
    self._model_config = configuration.ModelConfig()

  def _countModelParameters(self):
    """Counts the number of parameters in the model at top level scope."""
    counter = {}
    for v in tf.global_variables():
      name = v.op.name.split("/")[0]
      num_params = v.get_shape().num_elements()
      assert num_params
      counter[name] = counter.get(name, 0) + num_params
    return counter

  def _checkModelParameters(self):
    """Verifies the number of parameters in the model."""
    param_counts = self._countModelParameters()
    expected_param_counts = {
        "vgg_19": 143694632,
        # 2 * (context_size*hidden_size + hidden_size)
        "init_lstm": 525312,
        # vgg_output_size * context_size
        "context_encode": 262656,
        # hidden_size * context_size + context_size
        "attention_preactive": 262656,
        # context_size*1 + 1
        "attention_postactive": 513,
        # vocab_size * embedding_size
        "seq_embedding": 6144000,
        # hidden_size*1 + 1
        "selector": 513,
        # (embedding_size + num_lstm_units + 1) * 4 * num_lstm_units
        "lstm": 3147776,

        "logits": 262656,
        # (context_size + 1)*embeding_size
        "context_to_output": 262144,
        # (hidden_size + 1)*vocab_size
        "output": 6156000,
        "batch_norm": 12,
        "global_step": 1,
    }
    self.assertDictEqual(expected_param_counts, param_counts)

  def _checkOutputs(self, expected_shapes, feed_dict=None):
    """Verifies that the model produces expected outputs.

    Args:
      expected_shapes: A dict mapping Tensor or Tensor name to expected output
        shape.
      feed_dict: Values of Tensors to feed into Session.run().
    """
    fetches = expected_shapes.keys()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(fetches, feed_dict)

    for index, output in enumerate(outputs):
      tensor = fetches[index]
      expected = expected_shapes[tensor]
      actual = output.shape
      if expected != actual:
        self.fail("Tensor %s has shape %s (expected %s)." %
                  (tensor, actual, expected))

  def testBuildForTraining(self):
    model = ShowAttendTellModel(self._model_config, mode="train")
    model.build()

    self._checkModelParameters()

    expected_shapes = {
        # [batch_size, image_height, image_width, 3]
        model.images: (64, 224, 224, 3),
        # [batch_size, sequence_length]
        model.input_seqs: (64, 32),
        # [batch_size, sequence_length]
        model.target_seqs: (64, 32),
        # [batch_size, sequence_length]
        model.input_mask: (64, 32),
        # [batch_size, context_shape[0], context_shape[1]]
        model.context_encode: (64, 196, 512),
        # [batch_size, hidden_size]
        model.hidden: (64, 512),
        # [batch_size, hidden_size]
        model.c: (64, 512),
        # [batch_size, sequence_length, embedding_size]
        model.seq_embeddings: (64, 32, 512),
        # [batch_size, sequence_length, context_shape[0]]
        model.alphas: (64, 32, 196),
        # Scalar
        model.total_loss: (),
        # [batch_size * sequence_length]
        # model.target_cross_entropy_losses: (2048,),
    }
    self._checkOutputs(expected_shapes)

  def testBuildForEval(self):
    model = ShowAttendTellModel(self._model_config, mode="eval")
    model.build()

    self._checkModelParameters()

    expected_shapes = {
        # [batch_size, image_height, image_width, 3]
        model.images: (64, 224, 224, 3),
        # [batch_size, sequence_length]
        model.input_seqs: (64, 32),
        # [batch_size, sequence_length]
        model.target_seqs: (64, 32),
        # [batch_size, sequence_length]
        model.input_mask: (64, 32),
        # [batch_size, context_shape[0], context_shape[1]]
        model.context_encode: (64, 196, 512),
        # [batch_size, hidden_size]
        model.hidden: (64, 512),
        # [batch_size, hidden_size]
        model.c: (64, 512),
        # [batch_size, sequence_length, embedding_size]
        model.seq_embeddings: (64, 32, 512),
        # [batch_size, sequence_length, context_shape[0]]
        model.alphas: (64, 32, 196),
        # Scalar
        model.total_loss: (),
        # [batch_size * sequence_length]
        # model.target_cross_entropy_losses: (2048,),
    }
    self._checkOutputs(expected_shapes)

  def testBuildForInference(self):
    model = ShowAttendTellModel(self._model_config, mode="inference")
    model.build()

    self._checkModelParameters()

    # Test feeding an image to get the initial LSTM state.
    expected_shapes = {
        # [batch_size, image_height, image_width, 3]
        model.images: (64, 224, 224, 3),
        # [batch_size, sequence_length]
        model.input_seqs: (64, 1),
        # [batch_size, context_shape[0], context_shape[1]]
        model.context_encode: (64, 196, 512),
        # [batch_size, hidden_size]
        model.hidden: (64, 512),
        # [batch_size, hidden_size]
        model.c: (64, 512),
        # [batch_size, max_len]
        model.sampled_word_list: (64, 20),
        # [batch_size, max_len, context_shape[0]]
        model.alphas: (64, 20, 196),
    }
    self._checkOutputs(expected_shapes)


if __name__ == "__main__":
  tf.test.main()
