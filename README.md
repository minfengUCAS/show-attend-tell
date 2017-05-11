# show-attend-tell

A TensorFlow implementation of the image-to-text model descirbed in the paper:

"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"

Full text available at: https://arxiv.org/pdf/1502.03044.pdf

# Contact

**Author:** Minfeng Zhan

# Contents

# Model Overview

**Introduction**

The Show, attend and tell model is a deep neural network that learns how to describe the content of images. For example:


**Architecture**
The Show ,attend and tell is an example of attendion-based image caption generators under a common framework:
1) a "soft" deterministic attention mechanism trainable by standard back-propagation methods
2) a "hard" stochastic attention mechanism trainable by maximizing an approximate variational lower bound or equivalently by REINFORCE

We only imple the "soft" deterministic attendtion mechanism.

The following diagram illustates the model architecture.



# Getting Started
**Install Required Packages**
- Bazel
- TensorFlow 1.0 or greater
- NumPy
- Natural Language Toolkit(NLTK)

**Prepare the Training Data**
To train the model you will need to provide training data in native TFRecord format. The TFRecord format consists of a set of sharded fi
files containing serialized tf.SequenceExample protocol buffers. Each tf.SequenceExample proto contains an image (JPEG format), a caption
and metadata such as the image id.

Each caption is a list of words. During preprocessing, a directionary is created that assigns each word in the vocabulary to an integer-value
id. Each caption is encoded as a list of integer word ids in the tf.SequenceExample protos.

We have provided a script to download and preprocess the MSCOCO image caption data set into this format.

Before running the script, ensure that your hard disk has at least 150GB of available space for storing the downloaded and processed data/

```
# Location to save the MSCOCO data.
MSCOCO_DIR="${HOME}/im2txt/data/mscoco"

# Build the preprocessing script.
bazel build im2txt/download_and_preprocess_mscoco

# Run the preprocessing script.
bazel-bin/im2txt/download_and_preprocess_mscoco "${MSCOCO_DIR}"
```

**Download the VGG_19 Checkpoint**
The Show, attend and tell model requires a pretrained VGG_19 checkpoint file to initialize the parameters of its image encoder submodel

The checkpoint file is provided by the TensorFlow-Silm image classification library which provideds a suite of pre-trained image classification
models. You can read more about the models provided by the library here.

# Training a Model
**Initial Training**
Run the training script.
```
./train.sh
```

Run the evaluation script in a sparate process. This will log evaluation metrics to TensorBoard which allows training progress to be
monitored in real-time.

Note that you may run out of memory if you run the evaluation script on the same GPU as the training script. You can run the command 
export CUDA_VISIBLE_DEVICES="" to force the evaluation script to run on CPU. If evaluation runs too slowly on CPU, you can decrease the
value of --num_eval_examples.
```
./evaluation.sh
```

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
./TensorBoard.sh
```

# Generating Captions
Your trained Show, attend and tell model can generate captions for any JPEG image! The following command line will generate captions for
an image from the test set.
```
./caption_generator.sh
```

Example output:
Caption for image COCO_val2014_000000224477.jpg:
man surfing on a wave in the ocean.

Note: you may get different results. Some variation between different models is expected.

Here is the images.


