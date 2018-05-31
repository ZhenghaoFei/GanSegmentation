from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy

class ImageData:

    def __init__(self,
                 session,
                 image_paths,
                 batch_size,
                 load_size,
                 crop_size=256,
                 channels=3,
                 prefetch_batch=2,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=4096,
                 repeat=-1):

        self._sess = session
        self._img_batch = ImageData._image_batch(image_paths,
                                                 batch_size,
                                                 load_size,
                                                 crop_size,
                                                 channels,
                                                 prefetch_batch,
                                                 drop_remainder,
                                                 num_threads,
                                                 shuffle,
                                                 buffer_size,
                                                 repeat)
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    @staticmethod
    def _image_batch(image_paths,
                     batch_size,
                     load_size,
                     crop_size=256,
                     channels=3,
                     prefetch_batch=2,
                     drop_remainder=True,
                     num_threads=8,
                     shuffle=True,
                     buffer_size=4096,
                     repeat=-1):
        def _parse_func(path):
            img = tf.read_file(path)
            img = tf.image.decode_jpeg(img, channels=channels)
            # img = tf.image.random_flip_left_right(img)
            img = tf.image.resize_images(img, load_size)
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
            # img = tf.random_crop(img, [crop_size, crop_size, channels])
            # img = img * 2 - 1
            return img

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        dataset = dataset.map(_parse_func, num_parallel_calls=num_threads)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        if drop_remainder:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        else:
            dataset = dataset.batch(batch_size)

        dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

        return dataset.make_one_shot_iterator().get_next()



class ImageDataPair:

    def __init__(self,
                 session,
                 image_a_paths,
                 batch_size,
                 load_size=286,
                 crop_size=256,
                 channels=3,
                 prefetch_batch=2,
                 drop_remainder=True,
                 num_threads=8,
                 shuffle=True,
                 buffer_size=4096,
                 repeat=-1):

        self._sess = session
        self._img_batch = self._image_batch(image_a_paths,
                                                 batch_size,
                                                 load_size,
                                                 crop_size,
                                                 channels,
                                                 prefetch_batch,
                                                 drop_remainder,
                                                 num_threads,
                                                 shuffle,
                                                 buffer_size,
                                                 repeat,
                                                 match=True)

        self._img_batch_unmatch = self._image_batch(image_a_paths,
                                                 batch_size,
                                                 load_size,
                                                 crop_size,
                                                 channels,
                                                 prefetch_batch,
                                                 drop_remainder,
                                                 num_threads,
                                                 shuffle,
                                                 buffer_size,
                                                 repeat,
                                                 match=False)
        self._img_num = len(image_a_paths)

    def __len__(self):
        return self._img_num

    def batch_match(self):
        return self._sess.run(self._img_batch)


    def batch_unmatch(self):
        return self._sess.run(self._img_batch_unmatch)

    @staticmethod
    def _image_batch(image_a_paths,
                     batch_size,
                     load_size=286,
                     crop_size=256,
                     channels=3,
                     prefetch_batch=2,
                     drop_remainder=True,
                     num_threads=16,
                     shuffle=True,
                     buffer_size=4096,
                     repeat=-1,
                     match=True):

        def _parse_func(a_path, b_path):
            
            img_a = tf.read_file(a_path)
            img_b = tf.read_file(b_path)

            img_a = tf.image.decode_jpeg(img_a, channels=channels)
            img_b = tf.image.decode_jpeg(img_b, channels=1)

            img_a = tf.image.resize_images(img_a, load_size)
            img_a = (img_a - tf.reduce_min(img_a)) / (tf.reduce_max(img_a) - tf.reduce_min(img_a))
            # img_a = img_a * 2 - 1

            img_b = tf.image.resize_images(img_b, load_size)
            img_b = (img_b - tf.reduce_min(img_b)) / (tf.reduce_max(img_b) - tf.reduce_min(img_b))
            # img_b = img_b * 2 - 1


            img_pair = tf.concat([img_a, img_b], axis=2)
            return img_pair

        image_b_paths = copy.deepcopy(image_a_paths)

        if match:


            for i in range(len(image_b_paths)):
                image_b_paths[i] = image_b_paths[i].replace('A', 'B')
                image_b_paths[i] = image_b_paths[i].replace('_0', '_road_0')
        else:

            for i in range(len(image_b_paths)):
                image_b_paths[i] = image_b_paths[i].replace('A', 'B')
                image_b_paths[i] = image_b_paths[i].replace('_0', '_road_0')

            image_unpair_apaths = []
            image_unpair_bpaths = []

            for i in range(len(image_a_paths)):
                for j in range(len(image_b_paths)):
                    if i != j:
                        image_unpair_apaths.append(image_a_paths[i])
                        image_unpair_bpaths.append(image_b_paths[j])
            image_a_paths = image_unpair_apaths
            image_b_paths = image_unpair_bpaths

        dataset = tf.data.Dataset.from_tensor_slices((image_a_paths, image_b_paths))

        dataset = dataset.map(_parse_func, num_parallel_calls=num_threads)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size)

        dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

        return dataset.make_one_shot_iterator().get_next()

