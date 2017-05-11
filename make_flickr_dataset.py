import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *

vgg_model = '/home/minfeng.zhan/dataset/imageCaption/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/minfeng.zhan/dataset/imageCaption/VGG_ILSVRC_19_layers_deploy.prototxt'

annotation_path = '/home/minfeng.zhan/dataset/imageCaption/results_20130124.token'
flickr_image_path = '/home/minfeng.zhan/dataset/imageCaption/flickr30k-images/'
feat_path = './data/feats.npy'
annotation_result_path = './data/annotations.pickle'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
# annotations.drop([37028, 1416, 1417, 1418, 1419], inplace=True)

annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])

# annotations['img'] = annotations['image'].map(lambda x: x.split('#')[0])
# for i,img in enumerate(annotations['img']):
#     if img == '2623283166.jpg':
#         print(i)
#         print(img)

annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

unique_images = annotations['image'].unique()
image_df = pd.DataFrame({'image':unique_images, 'image_id':range(len(unique_images))})

annotations = pd.merge(annotations, image_df)
# print(annotations)
annotations.to_pickle(annotation_result_path)
#
# for img in unique_images:
#     print(img)
#     skimage.img_as_float(skimage.io.imread(img)).astype(np.float32)
if not os.path.exists(feat_path):
    feats = cnn.get_features(unique_images, layers='conv5_3', layer_sizes=[512,14,14])
    np.save(feat_path, feats)

