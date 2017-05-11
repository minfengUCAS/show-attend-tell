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

"""Image-to-text implementation based on Show, Attend and Tell.

"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov,
Richard S. Zemel, Yoshua Bengio
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from show_tell_attend.ops import image_embedding
from show_tell_attend.ops import image_processing
from show_tell_attend.ops import inputs as input_ops

class ShowAttendTellModel(object):
  """Image-to-text implementation based on Show, Attend and Tell.

  "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
  Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov,
  Richard S. Zemel, Yoshua Bengio
  """

  def __init__(self, config, mode, select=True, prev2out=True, ctx2out=True, alpha_c=0.0, dropout=True, train_vgg=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the VGG submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode

    self.train_vgg = train_vgg

    # sec4.2.1
    self.select = select

    # sec equ(2)
    self.ctx2out = ctx2out

    # sec squ(2)
    self.prev2out = prev2out

    # alpha_c
    self.alpha_c = alpha_c

    # whether use dropout or not
    self.dropout = dropout

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show Attend Tell" paper we initialize all variables with a
    # truncated_normal initializer.
    self.initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer = tf.constant_initializer(0.0)
    self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # An int32 recoder maximum padded_length
    self.caption_length = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the batch loss for the trainer to optimize.
    self.batch_loss = None
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # Collection of variables from the vgg submodel.
    self.vgg_variables = []

    # hidden state, c state
    # [batch_size, hidden_size]
    self.hidden = None
    # [batch_size, hidden_size]
    self.c = None

    # Context encode [batch_size, ]
    self.context = None
    self.context_encode = None

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    # Initial hidden W and b
    self.init_hidden_W = None
    self.init_hidden_b = None

    # Initial memory W and b
    self.init_memory_W = None
    self.init_memory_b = None

    # [batch_size, max_len]
    self.sampled_word_list = None

    # [vocab_size, embedding_size]
    self.embedding_map = None

    # [batch_size, caption_length, context_size]
    self.alphas = None

    # [batch_size, capiton_length]
    self.betas = None


  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def _batch_norm(self, x, mode='train'):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=0.95,
          center=True,
          scale=True,
          is_training=(mode == 'train'),
          updates_collections=None,
          scope='batch_norm'
      )


  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)

      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity,
                                           n_time_step=self.config.n_time_step))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask

  def _build_initial_lstm(self, mean_context):
      with tf.variable_scope("init_hidden") as scope:
        self.hidden = tf.contrib.layers.fully_connected(
          inputs=mean_context,
          num_outputs=self.config.hidden_size,
          activation_fn=tf.nn.tanh,
          weights_initializer=self.initializer,
          biases_initializer=self.const_initializer,
          scope=scope
        )
      with tf.variable_scope("init_c") as scope:
        self.c = tf.contrib.layers.fully_connected(
          inputs=mean_context,
          num_outputs=self.config.hidden_size,
          activation_fn=tf.nn.tanh,
          weights_initializer=self.initializer,
          biases_initializer=self.const_initializer,
          scope=scope
        )

  def build_context_encode(self):
    """Builds the image model subgraph and generates context_encode.

    Inputs:
      self.images

    Outputs:
      self.context_encode
    """
    vgg_output = image_embedding.vgg_19_extract(
        self.images,
        trainable=self.train_vgg,
        is_training=self.is_training())
    self.vgg_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_19")

    context = tf.reshape(vgg_output, [-1, self.config.context_shape[0], self.config.context_shape[1]])

    # Batch normalize feature vector
    if self.mode == "train":
      context = self._batch_norm(context, 'train')
    else:
      context = self._batch_norm(context, 'test')

    self.context = context

    self._build_initial_lstm(tf.reduce_mean(context, 1))

    # Map vgg output into embedding space.
    with tf.variable_scope("context_encode") as scope:
        context_flat = tf.reshape(context, [-1, self.config.context_size], name="context_flat")
        context_encode = tf.contrib.layers.fully_connected(
            inputs=context_flat,
            num_outputs=self.config.context_size,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)
        context_encode = tf.reshape(context_encode, [-1, self.config.context_shape[0], self.config.context_shape[1]])

    # Save the context_shape in the graph.
    tf.constant(self.config.context_size, name="context_encode_size")

    self.context_encode = context_encode

  def _bulid_attention(self, h, reuse=False):
    """Build the attention network

    :param hidden: lstm hidde state
    :return: self.alpha
    """
    with tf.variable_scope('attention_layer', reuse=reuse):
      w = tf.get_variable('w', [self.config.hidden_size, self.config.context_size], initializer=self.initializer)
      b = tf.get_variable('b', [self.config.context_size], initializer=self.const_initializer)
      h_att = tf.nn.relu(self.context_encode+tf.expand_dims(tf.matmul(h, w), 1) + b)

      w_att = tf.get_variable('w_att', [self.config.context_size, 1], initializer=self.initializer)
      out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.config.context_size]), w_att),
                           [-1, self.config.context_shape[0]])
      alpha = tf.nn.softmax(out_att)
      context = tf.reduce_sum(self.context * tf.expand_dims(alpha, 2), 1, name="context")
    return context, alpha


  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.emb_initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
    self.seq_embeddings = seq_embeddings
    self.embedding_map = embedding_map

  def _word_embedding(self, inputs):
    """Transform the word_idx into word_embedding

    :return: word embedding
    """

    x = tf.nn.embedding_lookup(self.embedding_map, inputs)

    return x

  def _selector(self, ctx_weight, hidden, reuse=False):
      with tf.variable_scope('selector', reuse=reuse) as scope:
          beta = tf.contrib.layers.fully_connected(
              inputs=hidden,
              num_outputs=1,
              activation_fn=tf.nn.sigmoid,
              weights_initializer=self.initializer,
              biases_initializer=self.const_initializer,
              scope=scope
          )
          ctx_weight = tf.multiply(beta, ctx_weight, name="selected_context")

          return ctx_weight, beta

  def _decode_lstm(self, embedding, h, ctx_weight, dropout=False, reuse=False):
      if dropout:
          h = tf.nn.dropout(h, self.config.dropout_keep_prob)

      with tf.variable_scope('logits', reuse=reuse) as logits:
          h_logits = tf.contrib.layers.fully_connected(
              inputs=h,
              num_outputs=self.config.embedding_size,
              activation_fn=None,
              weights_initializer=self.initializer,
              biases_initializer=self.const_initializer,
              scope=logits
          )

      if self.ctx2out:
          with tf.variable_scope('context_to_output') as cto:
              context2output = tf.contrib.layers.fully_connected(
                  inputs=ctx_weight,
                  num_outputs=self.config.embedding_size,
                  weights_initializer=self.initializer,
                  biases_initializer=None,
                  scope=cto
              )
              h_logits += context2output

      if self.prev2out:
          h_logits += embedding

      h_logits = tf.nn.tanh(h_logits)

      if dropout:
          h_logits = tf.nn.dropout(h_logits, self.config.dropout_keep_prob)

      with tf.variable_scope('output') as output:
          output_logits = tf.contrib.layers.fully_connected(
              inputs=h_logits,
              num_outputs=self.config.vocab_size,
              weights_initializer=self.initializer,
              biases_initializer=self.const_initializer,
              scope=output
          )

      return output_logits


  def build_model(self):
    """Builds the model.

    Inputs:
      self.context_encode
      self.seq_embeddings
      self.caption_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    batch_loss = 0.0
    alpha_list = []
    beta_list = []
    sampled_word_list = []

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)
    if self.mode == 'train':
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=self.config.lstm_dropout_keep_prob,
            output_keep_prob=self.config.lstm_dropout_keep_prob
        )
    h = self.hidden
    c = self.c
    x = self.seq_embeddings

    if self.mode == 'inference':
      for i in xrange(self.config.max_len):
        if i != 0:
          x = self._word_embedding(inputs=sampled_word)
        else:
          x = x[:, i, :]

        ctx_weight, alpha = self._bulid_attention(h, reuse=(i != 0))
        alpha_list.append(alpha)

        if self.select:
          ctx_weight, beta = self._selector(ctx_weight, h, reuse=(i != 0))
          beta_list.append(beta)

        with tf.variable_scope('lstm', reuse=(i != 0)):
          _, (c, h) = lstm_cell(inputs=tf.concat([x, ctx_weight], 1), state=[c, h])

        logits = self._decode_lstm(x, h, ctx_weight, reuse=(i != 0))
        sampled_word = tf.argmax(logits, 1)
        sampled_word_list.append(sampled_word)
      self.sampled_word_list = tf.transpose(sampled_word_list, (1, 0))

    else:

      for i in xrange(self.config.n_time_step):
        ctx_weight, alpha = self._bulid_attention(h, reuse=(i != 0))
        alpha_list.append(alpha)

        if self.select:
          ctx_weight, beta = self._selector(ctx_weight, h, reuse=(i != 0))
          beta_list.append(beta)

        with tf.variable_scope('lstm', reuse=(i != 0)):
          _, (c, h) = lstm_cell(inputs=tf.concat([x[:, i, :], ctx_weight], 1), state=[c, h])

        logits = self._decode_lstm(x[:, i, :], h, ctx_weight, dropout=self.dropout, reuse=(i != 0))
        batch_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_seqs[:, i],
                                                                             logits=logits)
                              * tf.to_float(self.input_mask[:, i]))

      if self.alpha_c > 0 and self.mode != 'inference':

        alphas = tf.transpose(alpha_list, (1, 0, 2))
        alphas_all = tf.reduce_sum(alphas, 1)
        alphas_reg = self.alpha_c * tf.reduce_sum((16.0/196 - alphas_all)**2)
        batch_loss += alphas_reg

      batch_loss /= tf.to_float(self.config.batch_size)
      tf.losses.add_loss(batch_loss)
      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
          tf.summary.histogram("parameters/" + var.op.name, var)
      self.batch_loss = batch_loss
      self.total_loss = total_loss
      self.target_cross_entropy_losses = batch_loss  # Used in evaluation.
    self.alphas = tf.transpose(alpha_list, (1, 0, 2))
    # self.betas = tf.transpose(tf.squeeze(beta_list), (1, 0))


  def setup_vgg_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.vgg_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring vgg variables from checkpoint file %s",
                        self.config.vgg_checkpoint_file)
        saver.restore(sess, self.config.vgg_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_context_encode()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_vgg_initializer()
    self.setup_global_step()
