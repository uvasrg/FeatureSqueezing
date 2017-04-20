import numpy as np

import tensorflow as tf
from tensorflow.python.platform import flags

from utils import load_externals
from cleverhans import utils_tf 
from cleverhans.utils_tf import batch_eval
from cleverhans.attacks import fgsm, jsma, jacobian_graph

FLAGS = flags.FLAGS

# Arguments for JSMA.
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')


def get_gradient_sign_tf(x, predictions):
    """
    TensorFlow implementation of calculting signed gradient with respect to x.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :return: a tensor for the adversarial example
    """

    # Compute loss
    y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss = utils_tf.tf_model_loss(y, predictions, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)
    signed_grad = tf.stop_gradient(signed_grad)
    return signed_grad


def generate_fgsm_examples(X_test, X_test_signed_gradient, eps_list):
    ret = {}
    for eps in eps_list:
        print ("Generating FGSM examples with epsilon %.1f" % eps)
        if eps == 0:
            ret[eps] = X_test
        else:
            X_test_adv = X_test + eps * X_test_signed_gradient
            ret[eps] = X_test_adv
    return ret


def calculate_signed_gradient_x(sess, x, predictions, X_test):
    signed_gradient = get_gradient_sign_tf(x, predictions)
    X_test_signed_gradient, = batch_eval(sess, [x], [signed_gradient], [X_test])
    return X_test_signed_gradient


def generate_jsma_examples(sess, x, predictions, X_test, Y_test, nb_examples):
    print('Crafting ' + str(nb_examples) + ' adversarial examples')

    # This array indicates whether an adversarial example was found for each
    # test set sample and target class
    results = np.zeros((FLAGS.nb_classes, nb_examples), dtype='i')

    # This array contains the fraction of perturbed features for each test set
    # sample and target class
    perturbations = np.zeros((FLAGS.nb_classes, nb_examples), dtype='f')

    # Define the TF graph for the model's Jacobian
    grads = jacobian_graph(predictions, x)

    X_adv_list = []
    perterb_list = []

    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(nb_examples):
        print ("=================Working on %d/%d..." % (sample_ind+1, nb_examples))
        # Define the target class.
        target = (int(np.argmax(Y_test[sample_ind])) + 1) % 10

        print('--------------------------------------')
        print('Creating adversarial example for target class ' + str(target))
        _X_adv, result, percentage_perterb = jsma(sess, x, predictions, grads,
                                             X_test[sample_ind:(sample_ind+1)],
                                             target, theta=1, gamma=0.1,
                                             increase=True, back='tf',
                                             clip_min=0, clip_max=1)

        # Update the arrays for later analysis
        results[target, sample_ind] = result
        perturbations[target, sample_ind] = percentage_perterb
        X_adv_list.append(_X_adv)
        perterb_list.append(percentage_perterb)

    X_adv = np.concatenate(tuple(X_adv_list))

    return X_adv, results, perterb_list