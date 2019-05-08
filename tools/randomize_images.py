#!/usr/bin/env python

import random, os, re
import tensorflow as tf
import numpy as np
import tfimage as im

in_dir = '/home/jordi/Desktop/cross-domain-disen/DATA/MNIST-CDCB-copy/train'
out_dir = '/home/jordi/Desktop/cross-domain-disen/DATA/MNIST-CDCB-copy/new_train'

if not os.path.exists(out_dir):
      os.makedirs(out_dir)

src_paths1 = []
src_paths2 = []
dst_paths = []

skipped = 0
for src_path in im.find(in_dir):
      name, _ = os.path.splitext(os.path.basename(src_path))
      dst_path = os.path.join(out_dir, name + ".png")
      if os.path.exists(dst_path):
            skipped += 1
      else:
            src_paths1.append(src_path)
            src_paths2.append(src_path)
            dst_paths.append(dst_path)

src_cutoff = len(src_paths1) / 2

src_paths1 = src_paths1[:, :src_cutoff]
src_paths2 = src_paths1[:, src_cutoff:]
dst_paths = dst_paths[:, :src_cutoff]


src_paths1.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
dst_paths.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

random.shuffle(src_paths2)


def split(src1,src2):

      # make sure that dimensions are correct
      height, width, _ = src1.shape

      width_cutoff = int(width / 2)

      # convert both images to RGB if necessary
      if src1.shape[2] == 1:
            src1 = im.grayscale_to_rgb(images=src1)
            src2 = im.grayscale_to_rgb(images=src2)

      # remove alpha channel
      if src2.shape[2] == 4:
            src1 = src1[:, :, :3]
            src2 = src2[:, :, :3]

      im1 = src1[:, :width_cutoff]
      im2 = src2[:, width_cutoff:]

      im_dest = np.concatenate([im1, im2], axis=1)

      return im_dest

with tf.Session() as sess:
      for src_path1, src_path2, dst_path_new in zip(src_paths1, src_paths2, dst_paths):

            srcA = im.load(src_path1)
            srcB = im.load(src_path2)
            new_im = split(srcA, srcB)
            #print(src_path1)
            #print(src_path2)
            #print(dst_path_new)
            test =  1
            im.save(new_im, dst_path_new)
