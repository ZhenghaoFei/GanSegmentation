from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
from glob import glob

import data
import image_utils as im
import models
import numpy as np
import tensorflow as tf
import utils
import cv2
import matplotlib.pyplot as plt

""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='cam2road', help='which dataset to use')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')

parser.add_argument('--input', dest='input', default='./', help='test images folder')
parser.add_argument('--model', dest='model', default='./', help='saved model folder')
parser.add_argument('--output', dest='output', default='./predict', help='prediction output folder')

args = parser.parse_args()

input_path = args.input
model = args.model
output_path = args.output

load_size = [256, 256]
crop_size = args.crop_size
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr


""" graph """
# a is always the real image and b is the segementation map

# models
generator_a2b = partial(models.generator1, scope='a2b')
discriminator_b = partial(models.discriminator, scope='b')

# operations
a_real = tf.placeholder(tf.float32, shape=[None, load_size[0], load_size[1], 3])
b_real = tf.placeholder(tf.float32, shape=[None, load_size[0], load_size[1], 1])

ab_match_pair = tf.placeholder(tf.float32, shape=[None, load_size[0], load_size[1], 4])
ab_unmatch_pair = tf.placeholder(tf.float32, shape=[None, load_size[0], load_size[1], 4])
ab_generator_pair = tf.placeholder(tf.float32, shape=[None, load_size[0], load_size[1], 4])


# generator
predict_op = generator_a2b(a_real)


""" train """
''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = utils.counter()


''' saver '''
saver = tf.train.Saver(max_to_keep=5)

''' restore '''
ckpt_dir = model
utils.load_checkpoint(ckpt_dir, sess)

''' data '''
a_test_img_paths = glob(input_path + '*')

# a_test_pool = data.ImageData(sess, a_test_img_paths, batch_size=3, load_size=load_size, shuffle=False, crop_size=crop_size)

ab_pair_test_pool = data.ImageDataPair(sess, a_test_img_paths, batch_size=3, load_size=load_size, shuffle=False, crop_size=crop_size)

ab_pair_ipt = ab_pair_test_pool.batch_match()

a_ipt = ab_pair_ipt[:,:,:,:3]
b_ipt = ab_pair_ipt[:,:,:,3:]

predict = sess.run(predict_op, feed_dict={a_real: a_ipt})
print('predict: ', predict.shape)

sample_opt = a_ipt
sample_opt[:,:,:,0] = 0.7*sample_opt[:,:,:,0] + 0.3* predict[:,:,:,0]


im.imwrite(im.immerge(sample_opt, len(sample_opt), 1), './result.jpg')

# test_img_paths = glob(input_path + '*')



# for path in test_img_paths:
#     img = tf.read_file(path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize_images(img, load_size)
#     img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
#     img = tf.expand_dims(img, axis=0)
#     img = sess.run(img)

#     predict_op = tf.image.resize_images(predict_op, [376, 1242])
#     predict = sess.run(predict_op, feed_dict={a_real: img})
#     predict = predict[0, :, :,0]
#     predict_uinit8 = (predict*255).astype('u1')

#     # save results
#     save_path = path.replace(input_path, output_path)
#     save_path = save_path.replace('_0', '_road_0')

#     cv2.imwrite(save_path, predict_uinit8)
#     print('saved ', save_path)

