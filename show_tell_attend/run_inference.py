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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np

import tensorflow as tf

from show_tell_attend import configuration
from show_tell_attend import inference_wrapper
from show_tell_attend.inference_utils import caption_generator
from show_tell_attend.inference_utils import vocabulary


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    inference = inference_wrapper.InferenceWrapper()
    restore_fn = inference.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)


    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = inference.caption_generator(sess, inference.model, image, np.array([vocab.start_id]))
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
     # Ignore begin and end words.
        caption = caption[0]
        idx = np.where(caption == vocab.end_id)
        sentence = [vocab.id_to_word(w) for w in caption[1:idx[0][0]]]
        sentence = " ".join(sentence)
        print("  %d) %s" % (i, sentence))


if __name__ == "__main__":
  tf.app.run()
