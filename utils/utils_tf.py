from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
import os
import keras
from keras.backend import categorical_crossentropy
import six
import tensorflow as tf
import time

from tensorflow.python.platform import flags
# from .utils import batch_indices

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

from utils import load_externals
from cleverhans.utils_tf import *

def tf_model_train_and_save(sess, x, y, predictions, X_train, Y_train, save_path='.', 
                   predictions_adv=None, evaluate=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: Boolean controling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :return: True if model trained
    """
    print("Starting model training using TensorFlow.")

    # Define loss
    loss = tf_model_loss(y, predictions)
    if predictions_adv is not None:
        loss = (loss + tf_model_loss(y, predictions_adv)) / 2

    train_step = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, rho=0.95, epsilon=1e-08).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
    print("Defined optimizer.")

    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)

        for epoch in six.moves.xrange(FLAGS.nb_epochs):
            print("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / FLAGS.batch_size))
            assert nb_batches * FLAGS.batch_size >= len(X_train)

            prev = time.time()
            for batch in range(nb_batches):

                # Compute batch start and end indices
                start, end = batch_indices(batch, len(X_train), FLAGS.batch_size)

                # Perform one training step
                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end],
                                          keras.backend.learning_phase(): 1})
            assert end >= len(X_train) # Check that all examples were used
            cur = time.time()
            print("\tEpoch took " + str(cur - prev) + " seconds")
            prev = cur
            if evaluate is not None:
                evaluate()


        if save_path:
            # save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Completed model training and model saved at:" + str(save_path))
        else:
            print("Completed model training.")

    return True

def tf_model_load_from_path(sess, save_path, vars):
    """

    :param sess:
    :param x:
    :param y:
    :param model:
    :return:
    """
    with sess.as_default():
        saver = tf.train.Saver(vars)
        saver.restore(sess, save_path)

    return True

def tf_model_eval_detect(sess, x, model1, model2, X_test):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :return: a float with the accuracy value
    """
    # Define sympbolic for accuracy
    # acc_value = keras.metrics.categorical_accuracy(y, model)

    l2_diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(model1, model2)),
                                    axis=1))
    l_inf_diff = tf.reduce_max(tf.abs(tf.sub(model1, model2)), axis=1)
    l1_diff = tf.reduce_sum(tf.abs(tf.sub(model1, model2)), axis=1)

    l2_mean_leg = 0.016
    l_inf_mean_leg = 0.014
    l1_mean_leg = 0.031

    def get_max_min_sum_leg_counts(l, threshold=0.016*5):
        l_max = tf.reduce_max(l)
        l_min = tf.reduce_min(l)
        l_sum = tf.reduce_sum(l)
        leg_counts = tf.count_nonzero(tf.less(l, tf.fill([FLAGS.batch_size], threshold)))
        return l_max, l_min, l_sum, leg_counts

    l_max, l_min, l_sum, leg_counts = get_max_min_sum_leg_counts(l1_diff, l1_mean_leg*15)

    # Init result var
    # accuracy = 0.0
    distance = 0.0
    distance_max = 0.0
    distance_min = 100.0
    leg_inputs = 0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= len(X_test)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * FLAGS.batch_size
            end = min(len(X_test), start + FLAGS.batch_size)
            cur_batch_size = end - start

            lmax_val, lmin_val, lsum_val, leg_counts_val = sess.run([l_max, l_min, l_sum, leg_counts], feed_dict={x: X_test[start:end],keras.backend.learning_phase(): 0})

            distance += lsum_val
            leg_inputs += leg_counts_val

            if distance_max < lmax_val:
                distance_max = lmax_val
            if distance_min > lmin_val:
                distance_min = lmin_val
            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
        assert end >= len(X_test)

        # Divide by number of examples to get final value
        distance /= len(X_test)
        leg_inputs /= float(len(X_test))

    return distance, distance_max, distance_min, leg_inputs

def tf_model_eval_distance(sess, x, model1, model2, X_test):
    """
    Compute the L1 distance between prediction of original and squeezed data.
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param model1: model output original predictions
    :param model2: model output squeezed predictions
    :param X_test: numpy array with training inputs
    :return: a float vector with the distance value
    """
    # Define sympbolic for accuracy
    # acc_value = keras.metrics.categorical_accuracy(y, model)

    l2_diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(model1, model2)),
                                    axis=1))
    l_inf_diff = tf.reduce_max(tf.abs(tf.sub(model1, model2)), axis=1)
    l1_diff = tf.reduce_sum(tf.abs(tf.sub(model1, model2)), axis=1)

    l1_dist_vec = np.zeros((len(X_test)))

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= len(X_test)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * FLAGS.batch_size
            end = min(len(X_test), start + FLAGS.batch_size)
            cur_batch_size = end - start

            l1_dist_vec[start:end] = l1_diff.eval(feed_dict={x: X_test[start:end],keras.backend.learning_phase(): 0})

        assert end >= len(X_test)
    return l1_dist_vec


def tf_model_eval_distance_dual_input(sess, x, model, X_test1, X_test2):
    """
    Compute the L1 distance between prediction of original and squeezed data.
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :return: a float with the accuracy value
    """
    # Define sympbolic for accuracy
    # acc_value = keras.metrics.categorical_accuracy(y, model)

    # l2_diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(model1, model2)),
    #                                 axis=1))
    # l_inf_diff = tf.reduce_max(tf.abs(tf.sub(model1, model2)), axis=1)
    # l1_diff = tf.reduce_sum(tf.abs(tf.sub(model1, model2)), axis=1)

    l1_dist_vec = np.zeros((len(X_test1)))

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test1)) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= len(X_test1)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * FLAGS.batch_size
            end = min(len(X_test1), start + FLAGS.batch_size)
            cur_batch_size = end - start

            pred_1 = model.eval(feed_dict={x: X_test1[start:end],keras.backend.learning_phase(): 0})
            pred_2 = model.eval(feed_dict={x: X_test2[start:end],keras.backend.learning_phase(): 0})

            l1_dist_vec[start:end] = np.sum(np.abs(pred_1 - pred_2), axis=1)
        assert end >= len(X_test1)

    return l1_dist_vec


def tf_model_eval_dist_tri_input(sess, x, model, X_test1, X_test2, X_test3, mode = 'max'):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param model: model output predictions
    :param X_test[1,2,3]: numpy array with testing inputs
    :param Y_test: numpy array with training outputs
    :return: a float with the accuracy value
    """

    l1_dist_vec = np.zeros((len(X_test1)))

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test1)) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= len(X_test1)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * FLAGS.batch_size
            end = min(len(X_test1), start + FLAGS.batch_size)
            cur_batch_size = end - start

            pred_1 = model.eval(feed_dict={x: X_test1[start:end],keras.backend.learning_phase(): 0})
            pred_2 = model.eval(feed_dict={x: X_test2[start:end],keras.backend.learning_phase(): 0})
            pred_3 = model.eval(feed_dict={x: X_test3[start:end],keras.backend.learning_phase(): 0})

            l11 = np.sum(np.abs(pred_1 - pred_2), axis=1)
            l12 = np.sum(np.abs(pred_1 - pred_3), axis=1)
            l13 = np.sum(np.abs(pred_2 - pred_3), axis=1)

            if mode == 'max':
                l1_dist_vec[start:end] = np.max(np.array([l11, l12, l13]), axis=0)
            elif mode == 'mean':
                l1_dist_vec[start:end] = np.mean(np.array([l11, l12, l13]), axis=0)
        assert end >= len(X_test1)

        # Divide by number of examples to get final value

    return l1_dist_vec