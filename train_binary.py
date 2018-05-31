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

import matplotlib.pyplot as plt

""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='cam2road', help='which dataset to use')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--saveidx', dest='saveidx', default='0', help='appendix')

args = parser.parse_args()

dataset = args.dataset
saveidx = args.saveidx

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
a2b = generator_a2b(a_real)

aa2b_pair = tf.concat([a_real, a2b], axis=3, name='aa2b_pair')

# discriminator
a2b_logit = discriminator_b(aa2b_pair)

ab_match_logit = discriminator_b(ab_match_pair)
ab_unmatch_logit = discriminator_b(ab_unmatch_pair)
a2b_sample_logit = discriminator_b(ab_generator_pair)

# losses
# generative loss
g_loss_a2b = tf.losses.mean_squared_error(a2b_logit, tf.ones_like(a2b_logit))
ce_loss = tf.losses.absolute_difference(a2b, b_real)
g_loss = g_loss_a2b

# discriminative loss
d_loss_ab_match = tf.losses.mean_squared_error(ab_match_logit, tf.ones_like(ab_match_logit))
d_loss_ab_unmatch = tf.losses.mean_squared_error(ab_unmatch_logit, tf.zeros_like(ab_unmatch_logit))
d_loss_aa2b_pair = tf.losses.mean_squared_error(a2b_sample_logit, tf.zeros_like(a2b_sample_logit))

d_loss_b = d_loss_ab_match + d_loss_ab_unmatch + d_loss_aa2b_pair

# summaries
g_summary = utils.summary({g_loss_a2b: 'g_loss_a2b'})
d_summary_b = utils.summary({d_loss_b: 'd_loss_b'})

# optim
t_var = tf.trainable_variables()
d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]

d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)


""" train """
''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = utils.counter()

''' data '''
a_img_paths = glob('./datasets/' + dataset + '/trainA/*')
b_img_paths = glob('./datasets/' + dataset + '/trainB/*')

ab_pair_data_pool = data.ImageDataPair(sess, a_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

a_data_pool = data.ImageData(sess, a_img_paths, batch_size, load_size=load_size, crop_size=crop_size, channels=3)
b_data_pool = data.ImageData(sess, b_img_paths, batch_size, load_size=load_size, crop_size=crop_size, channels=1)

a_test_img_paths = glob('./datasets/' + dataset + '/testA/*')
b_test_img_paths = glob('./datasets/' + dataset + '/testB/*')

a_test_pool = data.ImageData(sess, a_test_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

a2b_pool = utils.ItemPool()

# # #########################
# # PLOT
# a_real_ipt = a_data_pool.batch()
# b_real_ipt = b_data_pool.batch()

# ab_match_pair_ipt = ab_pair_data_pool.batch_match()

# img = ab_match_pair_ipt[0]
# print('img', img.shape)

# label = np.zeros_like(img[:, :, :3])
# label[:,:,0] = img[:, :, 3]

# print('label', label.shape)
# img = np.concatenate((img[:, :, :3], label), axis=1)
# print('img', img.shape)
# img = im._im2uint(img)

# plt.imshow(img)
# plt.show()


# img = a_real_ipt[0]
# img = im._im2uint(img)
# plt.imshow(img)
# plt.show()



# img = np.zeros_like(a_real_ipt)
# print('img', img.shape)
# print('a_real_ipt', a_real_ipt.shape)
# print('b_real_ipt', b_real_ipt.shape)

# img[:,:,:,0] = b_real_ipt[:,:,:,0]

# img = img[0,:,:,:]

# img = im._im2uint(img)
# plt.imshow(img)
# plt.show()

# # #########################

''' summary '''
summary_writer = tf.summary.FileWriter('./outputs/summaries/' + dataset + saveidx, sess.graph)

''' saver '''
saver = tf.train.Saver(max_to_keep=5)

''' restore '''
ckpt_dir = './outputs/checkpoints/' + dataset + saveidx
utils.mkdir(ckpt_dir)
try:
    utils.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

'''train'''
# try:
batch_epoch = min(len(a_data_pool), len(b_data_pool)) // batch_size
max_it = epoch * batch_epoch
for it in range(sess.run(it_cnt), max_it):
    sess.run(update_cnt)
    epoch = it // batch_epoch
    it_epoch = it % batch_epoch + 1

    # prepare data
    a_real_ipt = a_data_pool.batch()
    b_real_ipt = b_data_pool.batch()

    a2b_opt = sess.run(a2b, feed_dict={a_real: a_real_ipt})
    a2b_pair_opt = np.concatenate((a_real_ipt, a2b_opt), axis=3)

    a2b_sample_ipt = np.array(a2b_pool(list(a2b_pair_opt)))

    # train G
    g_summary_opt, _ = sess.run([g_summary, g_train_op], feed_dict={a_real: a_real_ipt})
    summary_writer.add_summary(g_summary_opt, it)

    # train D_b
    ab_match_pair_ipt = ab_pair_data_pool.batch_match()
    ab_unmatch_pair_ipt = ab_pair_data_pool.batch_unmatch()

    d_summary_b_opt, _ = sess.run([d_summary_b, d_b_train_op], 
                                    feed_dict={b_real: b_real_ipt, 
                                               ab_generator_pair: a2b_sample_ipt,
                                               ab_match_pair: ab_match_pair_ipt,
                                               ab_unmatch_pair: ab_unmatch_pair_ipt
                                               })
    summary_writer.add_summary(d_summary_b_opt, it)


    # display
    if it % 1 == 0:
        print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

    # save
    if (it + 1) % 1000 == 0:
        save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
        print('Model saved in file: % s' % save_path)

    # sample
    if (it + 1) % 100 == 0:
        a_real_ipt = a_test_pool.batch()
        a2b_opt = sess.run(a2b, feed_dict={a_real: a_real_ipt})

        print('a_real_ipt', a_real_ipt.shape)
        print('a2b_opt', a2b_opt.shape)

        # label = np.zeros_like(a_real_ipt)
        # label[:,:,:,0] = a2b_opt[:,:,:,0]
        
        # sample_opt = np.concatenate((a_real_ipt, label), axis=0)
        sample_opt = a_real_ipt
        sample_opt[:,:,:,0] = 0.7*sample_opt[:,:,:,0] + 0.3* a2b_opt[:,:,:,0]

        save_dir = './outputs/sample_images_while_training/' + dataset + saveidx
        utils.mkdir(save_dir)
        im.imwrite(im.immerge(sample_opt, 1, 1), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))
# except:
#     print("ERROR")
#     save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
#     print('Model saved in file: % s' % save_path)
#     sess.close()
