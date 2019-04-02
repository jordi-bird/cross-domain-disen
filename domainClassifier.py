from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

from ops import *

@tf.RegisterGradient("ReverseGradClassifier")
def _reverse_grad(unused_op, grad):
    return -1.0*grad


def create_domain_classifier(sR, a):
    initial_input = flatten(sR)

    # decoder_3: [batch, 8, 8, ngf] => [batch, 1, 1, 100]
    # decoder_2: [batch, 1, 1, 100] => [batch, 1, 1, 1]

    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": "ReverseGradClassifier"}):
        input = tf.identity(initial_input)

    with tf.variable_scope("class_layer_100"):
        fc_100 = discrim_fc(input, 100)
        rectified = lrelu(fc_100, 0.2)

    with tf.variable_scope("class_layer_1"):
        fc_1 = discrim_fc(rectified, out_channels=1)

    return fc_1
