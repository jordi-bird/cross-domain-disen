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
from encoder import *
from decoder import *
from decoderExclusive import *
from domainClassifier import *
from discriminatorWGANGP import *


LAMBDA = 10

Model = collections.namedtuple("Model", "outputsX2Y, outputsY2X,\
                               outputsX2Yp, outputsY2Xp,\
                               discrim_sharedX2Y_loss,discrim_sharedY2X_loss,\
                               predict_realX2Y, predict_realY2X,\
                               predict_fakeX2Y, predict_fakeY2X,\
                               sR_X2Y,sR_Y2X,\
                               eR_X2Y,eR_Y2X,\
                               discrimX2Y_loss, discrimY2X_loss,\
                               genX2Y_loss, genY2X_loss,\
                               cycX_output,cycX_loss,\
                               cycY_output,cycY_loss,\
                               code_recon_loss,\
                               code_sR_X2Y_recon_loss,code_sR_Y2X_recon_loss,\
                               code_eR_X2Y_recon_loss,code_eR_Y2X_recon_loss,\
                               im_swapped_Y,sel_auto_Y\
                               im_swapped_X,sel_auto_X\
                               train")

def create_model(inputsX, inputsY, a):

    # Modify values if images are reduced
    IMAGE_SIZE = 256

    OUTPUT_DIM = IMAGE_SIZE*IMAGE_SIZE*3 # 256x256x3

    # Target for inputsX is inputsY and vice versa
    targetsX = inputsY
    targetsY = inputsX

    ######### IMAGE_TRANSLATORS
    with tf.variable_scope("generatorX2Y_encoder"):
        sR_X2Y, eR_X2Y = create_generator_encoder(inputsX, a)

    with tf.variable_scope("generatorY2X_encoder"):
        sR_Y2X, eR_Y2X = create_generator_encoder(inputsY, a)

    # Generate random noise to substitute exclusive rep
    z = tf.random_normal(eR_X2Y.shape)
    z2 = tf.random_normal(eR_X2Y.shape)

    # One copy of the decoder for the noise input, the second copy for the correct the cross-domain autoencoder
    with tf.name_scope("generatorX2Y_decoder_noise"):
        with tf.variable_scope("generatorX2Y_decoder"):
            out_channels = int(targetsX.get_shape()[-1])
            outputsX2Y = create_generator_decoder(sR_X2Y, z, out_channels, a)

        with tf.variable_scope("generatorX2Y_decoder", reuse=True):
            outputsX2Yp = create_generator_decoder(sR_X2Y, z2, out_channels, a)

    with tf.name_scope("generatorX2Y_reconstructor"):
        with tf.variable_scope("generatorY2X_encoder", reuse=True):
            sR_X2Y_recon, eR_X2Y_recon = create_generator_encoder(outputsX2Y, a)

    #CYCLE-CONSISTENCY
    with tf.name_scope("generatorX2Y_cyc"):
        with tf.variable_scope("generatorX2Y_decoder", reuse=True):
            outputX2Y_cyc = create_generator_decoder(sR_X2Y_recon, eR_X2Y, out_channels, a)

    with tf.name_scope("generatorY2X_decoder_noise"):
        with tf.variable_scope("generatorY2X_decoder"):
            out_channels = int(targetsY.get_shape()[-1])
            outputsY2X = create_generator_decoder(sR_Y2X, z, out_channels, a)

        with tf.variable_scope("generatorY2X_decoder", reuse=True):
            outputsY2Xp = create_generator_decoder(sR_Y2X, z2, out_channels, a)

    with tf.name_scope("generatorY2X_reconstructor"):
        with tf.variable_scope("generatorX2Y_encoder", reuse=True):
            sR_Y2X_recon, eR_Y2X_recon = create_generator_encoder(outputsY2X, a)

    # CYCLE-CONSISTENCY
    with tf.name_scope("generatorY2X_cyc"):
        with tf.variable_scope("generatorY2X_decoder", reuse=True):
            outputY2X_cyc = create_generator_decoder(sR_Y2X_recon, eR_Y2X, out_channels, a)


    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables

    # We will now have 2 different discriminators, one per direction, and two
    # copies of each for real/fake pairs

    with tf.name_scope("real_discriminatorX2Y"):
        with tf.variable_scope("discriminatorX2Y"):
            predict_realX2Y = create_discriminator(inputsX, targetsX, a)

    with tf.name_scope("real_discriminatorY2X"):
        with tf.variable_scope("discriminatorY2X"):
            predict_realY2X = create_discriminator(inputsY, targetsY, a)

    with tf.name_scope("fake_discriminatorX2Y"):
        with tf.variable_scope("discriminatorX2Y", reuse=True):
            predict_fakeX2Y = create_discriminator(inputsX, outputsX2Y, a)

    with tf.name_scope("fake_discriminatorY2X"):
        with tf.variable_scope("discriminatorY2X", reuse=True):
            predict_fakeY2X = create_discriminator(inputsY, outputsY2X, a)

    ######### VISUAL ANALOGIES
    # This is only for visualization (visual analogies), not used in training loss
    with tf.name_scope("image_swapper_X"):
        im_swapped_X,sel_auto_X = create_visual_analogy(sR_X2Y, eR_X2Y,
                                                 outputY2X_cyc,inputsX,'Y2X', a)
    with tf.name_scope("image_swapper_Y"):
        im_swapped_Y,sel_auto_Y = create_visual_analogy(sR_Y2X, eR_Y2X,
                                                  outputX2Y_cyc,inputsY,'X2Y', a)


    ######### SHARED REPRESENTATION
    # Create generators/discriminators for exclusive representation

    with tf.name_scope("discriminator_sharedX2Y"):
        with tf.variable_scope("discriminator_sharedX2Y"):
            predict_fake_sharedX2Y = create_domain_classifier(sR_X2Y, a)

    with tf.name_scope("discriminator_sharedY2X"):
        with tf.variable_scope("discriminator_sharedY2X"):
            predict_fake_sharedY2X = create_domain_classifier(sR_Y2X, a)


    ######### LOSSES

    with tf.name_scope("generatorX2Y_loss"):
        genX2Y_loss_GAN = -tf.reduce_mean(predict_fakeX2Y)
        genX2Y_loss = genX2Y_loss_GAN * a.gan_weight

    with tf.name_scope("discriminatorX2Y_loss"):
        discrimX2Y_loss = tf.reduce_mean(predict_fakeX2Y) - tf.reduce_mean(predict_realX2Y)
        alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
        differences = tf.reshape(outputsX2Y,[-1,OUTPUT_DIM])-tf.reshape(targetsX,[-1,OUTPUT_DIM])
        interpolates = tf.reshape(targetsX, [-1,OUTPUT_DIM]) + (alpha*differences)
        with tf.variable_scope("discriminatorX2Y", reuse=True):
            gradients = tf.gradients(create_discriminator(inputsX,tf.reshape(interpolates,[-1,IMAGE_SIZE,IMAGE_SIZE,3]),a),
                         [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        tf.summary.histogram("X2Y/fake_score", predict_fakeX2Y)
        tf.summary.histogram("X2Y/real_score", predict_realX2Y)
        tf.summary.histogram("X2Y/disc_loss", discrimX2Y_loss )
        tf.summary.histogram("X2Y/gradient_penalty", gradient_penalty)
        discrimX2Y_loss += LAMBDA*gradient_penalty

    with tf.name_scope("generatorY2X_loss"):
        genY2X_loss_GAN = -tf.reduce_mean(predict_fakeY2X)
        genY2X_loss = genY2X_loss_GAN * a.gan_weight

    with tf.name_scope("discriminatorY2X_loss"):
        discrimY2X_loss = tf.reduce_mean(predict_fakeY2X) - tf.reduce_mean(predict_realY2X)
        alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
        differences = tf.reshape(outputsY2X,[-1,OUTPUT_DIM])-tf.reshape(targetsY,[-1,OUTPUT_DIM])
        interpolates = tf.reshape(targetsY,[-1,OUTPUT_DIM]) + (alpha*differences)
        with tf.variable_scope("discriminatorY2X", reuse=True):
            gradients = tf.gradients(create_discriminator(inputsY,tf.reshape(interpolates,[-1,IMAGE_SIZE,IMAGE_SIZE,3]),a),
                         [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrimY2X_loss += LAMBDA*gradient_penalty

    #SHARED GRL LOSS

    with tf.name_scope("discriminator_sharedX2Y_loss"):
        labels_X2Y = tf.zeros([a.batch_size, 1], dtype=tf.float32)
        cross_entropyX2Y = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_X2Y, logits=predict_fake_sharedX2Y)
        discrim_sharedX2Y_loss = tf.reduce_mean(cross_entropyX2Y)
        discrim_sharedX2Y_loss = discrim_sharedX2Y_loss * a.classifier_shared_weight



    with tf.name_scope("discriminator_sharedY2X_loss"):
        labels_Y2X = tf.ones([a.batch_size, 1], dtype=tf.float32)
        cross_entropyY2X = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_Y2X, logits=predict_fake_sharedY2X)
        discrim_sharedY2X_loss = tf.reduce_mean(cross_entropyY2X)
        discrim_sharedY2X_loss = discrim_sharedY2X_loss * a.classifier_shared_weight


    with tf.name_scope("code_recon_loss"):
        code_sR_X2Y_recon_loss = tf.reduce_mean(tf.abs(sR_X2Y_recon-sR_X2Y))
        code_sR_Y2X_recon_loss = tf.reduce_mean(tf.abs(sR_Y2X_recon-sR_Y2X))
        code_eR_X2Y_recon_loss = tf.reduce_mean(tf.abs(eR_X2Y_recon-z))
        code_eR_Y2X_recon_loss = tf.reduce_mean(tf.abs(eR_Y2X_recon-z))
        code_recon_loss = a.l1_weight*(code_sR_X2Y_recon_loss + code_sR_Y2X_recon_loss
                                    +code_eR_X2Y_recon_loss + code_eR_Y2X_recon_loss)

    #CYCLE-CONSISTENCY LOSS
    with tf.name_scope("cycX_loss"):
        cycX_loss = a.l1_weight*tf.reduce_mean(tf.abs(outputX2Y_cyc-inputsX))

    with tf.name_scope("cycY_loss"):
        cycY_loss = a.l1_weight*tf.reduce_mean(tf.abs(outputY2X_cyc-inputsY))

    ######### OPTIMIZERS

    with tf.name_scope("discriminatorX2Y_train"):
        discrimX2Y_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorX2Y")]
        discrimX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrimX2Y_grads_and_vars = discrimX2Y_optim.compute_gradients(discrimX2Y_loss, var_list=discrimX2Y_tvars)
        discrimX2Y_train = discrimX2Y_optim.apply_gradients(discrimX2Y_grads_and_vars)

    with tf.name_scope("discriminatorY2X_train"):
        discrimY2X_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorY2X")]
        discrimY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrimY2X_grads_and_vars = discrimY2X_optim.compute_gradients(discrimY2X_loss, var_list=discrimY2X_tvars)
        discrimY2X_train = discrimY2X_optim.apply_gradients(discrimY2X_grads_and_vars)

    with tf.name_scope("generatorX2Y_train"):
        with tf.control_dependencies([discrimX2Y_train]):
            genX2Y_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorX2Y")]
            genX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            genX2Y_grads_and_vars = genX2Y_optim.compute_gradients(genX2Y_loss, var_list=genX2Y_tvars)
            genX2Y_train = genX2Y_optim.apply_gradients(genX2Y_grads_and_vars)

    with tf.name_scope("generatorY2X_train"):
        with tf.control_dependencies([discrimY2X_train]):
            genY2X_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorY2X")]
            genY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            genY2X_grads_and_vars = genY2X_optim.compute_gradients(genY2X_loss, var_list=genY2X_tvars)
            genY2X_train = genY2X_optim.apply_gradients(genY2X_grads_and_vars)


    #SHARED GRL OPTIMIZATION
    with tf.name_scope("discriminator_sharedX2Y_train"):
        discrim_sharedX2Y_tvars = [var for var in tf.trainable_variables() if
                                      var.name.startswith("discriminator_sharedX2Y")]
        discrim_sharedX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_sharedX2Y_grads_and_vars = discrim_sharedX2Y_optim.compute_gradients(discrim_sharedX2Y_loss,
                                                                                           var_list=discrim_sharedX2Y_tvars)
        discrim_sharedX2Y_train = discrim_sharedX2Y_optim.apply_gradients(discrim_sharedX2Y_grads_and_vars)

    with tf.name_scope("discriminator_sharedY2X_train"):
        discrim_sharedY2X_tvars = [var for var in tf.trainable_variables() if
                                      var.name.startswith("discriminator_sharedY2X")]
        discrim_sharedY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_sharedY2X_grads_and_vars = discrim_sharedY2X_optim.compute_gradients(discrim_sharedY2X_loss,
                                                                                           var_list=discrim_sharedY2X_tvars)
        discrim_sharedY2X_train = discrim_sharedY2X_optim.apply_gradients(discrim_sharedY2X_grads_and_vars)


    with tf.name_scope("code_recon_train"):
        code_recon_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y") or
                              var.name.startswith("generatorY2X")]
        code_recon_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        code_recon_grads_and_vars = code_recon_optim.compute_gradients(code_recon_loss, var_list=code_recon_tvars)
        code_recon_train = code_recon_optim.apply_gradients(code_recon_grads_and_vars)

    #CYCLE-CONSISTENCY OPTIMIZATION
    with tf.name_scope("generatorX2Y_cyc_train"):
        cycX_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y_encoder") or
                              var.name.startswith("generatorX2Y_decoder") or
                              var.name.startswith("generatorY2X_encoder") or
                              var.name.startswith("generatorY2X_decoder")]
        cycX_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        cycX_grads_and_vars = cycX_optim.compute_gradients(cycX_loss, var_list=cycX_tvars)
        cycX_train = cycX_optim.apply_gradients(cycX_grads_and_vars)

    with tf.name_scope("generatorY2X_cyc_train"):
        cycY_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y_encoder") or
                              var.name.startswith("generatorX2Y_decoder") or
                              var.name.startswith("generatorY2X_encoder") or
                              var.name.startswith("generatorY2X_decoder")]
        cycY_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        cycY_grads_and_vars = cycY_optim.compute_gradients(cycY_loss, var_list=cycY_tvars)
        cycY_train = cycY_optim.apply_gradients(cycY_grads_and_vars)




    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrimX2Y_loss, discrimY2X_loss,
                               genX2Y_loss, genY2X_loss,
                               code_recon_loss,
                               code_sR_X2Y_recon_loss, code_sR_Y2X_recon_loss,
                               code_eR_X2Y_recon_loss, code_eR_Y2X_recon_loss,
                               discrim_sharedX2Y_loss, discrim_sharedY2X_loss,
                               cycX_loss, cycY_loss])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    return Model(
        predict_realX2Y=predict_realX2Y,
        predict_realY2X=predict_realY2X,
        predict_fakeX2Y=predict_fakeX2Y,
        predict_fakeY2X=predict_fakeY2X,
        im_swapped_X=im_swapped_X,
        im_swapped_Y=im_swapped_Y,
        sel_auto_X=sel_auto_X,
        sel_auto_Y=sel_auto_Y,
        sR_X2Y=sR_X2Y,
        sR_Y2X=sR_Y2X,
        eR_X2Y=eR_X2Y,
        eR_Y2X=eR_Y2X,
        discrimX2Y_loss=ema.average(discrimX2Y_loss),
        discrimY2X_loss=ema.average(discrimY2X_loss),
        genX2Y_loss=ema.average(genX2Y_loss),
        genY2X_loss=ema.average(genY2X_loss),
        discrim_sharedX2Y_loss=ema.average(discrim_sharedX2Y_loss),
        discrim_sharedY2X_loss=ema.average(discrim_sharedY2X_loss),
        outputsX2Y=outputsX2Y,
        outputsY2X=outputsY2X,
        outputsX2Yp=outputsX2Yp,
        outputsY2Xp=outputsY2Xp,
        cycX_output=outputX2Y_cyc,
        cycX_loss=ema.average(cycX_loss),
        cycY_output=outputY2X_cyc,
        cycY_loss=ema.average(cycY_loss),
        code_recon_loss=ema.average(code_recon_loss),
        code_sR_X2Y_recon_loss=ema.average(code_sR_X2Y_recon_loss),
        code_sR_Y2X_recon_loss=ema.average(code_sR_Y2X_recon_loss),
        code_eR_X2Y_recon_loss=ema.average(code_eR_X2Y_recon_loss),
        code_eR_Y2X_recon_loss=ema.average(code_eR_Y2X_recon_loss),
        train=tf.group(update_losses, incr_global_step, genX2Y_train,
                       genY2X_train,code_recon_train,
                       discrim_sharedX2Y_train,discrim_sharedY2X_train,
                       cycX_train, cycY_train),
    )


def create_visual_analogy(sR, eR, auto_output, inputs, which_direction, a):
        swapScoreBKG = 0
        sR_Swap = []
        eR_Swap = []
        sel_auto = []

        for i in range(0,a.batch_size):
            s_curr = tf.reshape(sR[i,:],[sR.shape[1],sR.shape[2],sR.shape[3]])

            # Take a random image from the batch, make sure it is different from current
            bkg_ims_idx = random.randint(0,a.batch_size-1)
            while bkg_ims_idx == i:
                bkg_ims_idx = random.randint(0,a.batch_size-1)

            ex_rnd = tf.reshape(eR[bkg_ims_idx,:],[eR.shape[1]])
            sR_Swap.append(s_curr)
            eR_Swap.append(ex_rnd)

            # Store also selected reference image for visualization
            sel_auto.append(inputs[bkg_ims_idx,:])

        with tf.variable_scope("generator" + which_direction + "_decoder", reuse=True):
                    out_channels = int(auto_output.get_shape()[-1])
                    im_swapped = create_generator_decoder(tf.stack(sR_Swap),
                                                          tf.stack(eR_Swap), out_channels, a)



        return im_swapped, tf.stack(sel_auto)


