from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import pdb
import keras
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils import load_externals
from cleverhans.utils_mnist import model_mnist
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval

from utils.utils_tf import tf_model_train_and_save, tf_model_load_from_path
from utils.utils_tf import tf_model_eval_detect, tf_model_eval_distance, tf_model_eval_distance_dual_input, tf_model_eval_dist_tri_input
from utils.load_data import maybe_download_mnist_model, restore_or_calculate_object
from utils.attacks import calculate_signed_gradient_x, generate_fgsm_examples, generate_jsma_examples
from utils.result_processing import write_to_csv
from utils.visualization import draw_fgsm_adv_examples, draw_jsma_adv_examples
from utils.detection import train_detector

from train_mnist_model import load_tf_session, get_mnist_data
from squeeze import binary_filter_tf, reduce_precision_tf, median_filter_np, binary_filter_np

FLAGS = flags.FLAGS

# Arguments for task scheduling
flags.DEFINE_boolean('visualize', False, 'Output the image examples as image or not.')
flags.DEFINE_string('task', '?', 'Supported tasks: FGSM, JSMA, FGSM-detection, JSMA-detection, joint-detection.')


def get_fgsm_adv_examples(sess, x, predictions, X_test, eps_list, model_name, nb_examples, result_folder):
    # Generate or load FGSM adversarial examples.
    X_test = X_test[:nb_examples,:]
    x_signed_gradient_fpath = model_name + '_x_test_signed_gradient_%dexamples.pickle' % nb_examples
    x_signed_gradient_fpath = os.path.join(result_folder, x_signed_gradient_fpath)
    func = calculate_signed_gradient_x
    args = (sess, x, predictions, X_test)
    X_test_signed_gradient = restore_or_calculate_object(x_signed_gradient_fpath, func, args, 'Signed Gradients wrt X')

    adv_x_dict = generate_fgsm_examples(X_test, X_test_signed_gradient, eps_list)
    return adv_x_dict

def get_jsma_adv_examples(sess, x, predictions, X_test, Y_test, model_name, nb_examples, result_folder):
    adv_x_fname = model_name + '_x_test_adv_jsma_%dexamples.pickle' % nb_examples
    adv_x_fpath = os.path.join(result_folder, adv_x_fname)
    func = generate_jsma_examples
    args = (sess, x, predictions, X_test, Y_test, nb_examples)
    X_adv, results, perterb_list = restore_or_calculate_object(adv_x_fpath, func, args, obj_name = "JSMA adversarial examples")

    print ("\n===Evaluating the success rate of JSMA-adversarial examples...")
    success_rate = float(np.sum(results)) / nb_examples
    print('Avg. rate of successful misclassifcations {0}'.format(success_rate))
    # Compute the average distortion introduced by the algorithm
    percentage_perturbed = np.mean(perterb_list)
    print('Avg. rate of perterbed features {0}'.format(percentage_perturbed))
    print ("---Done.")
    return X_adv

def calculate_accuracy_adv_fgsm(sess, x, y, predictions, predictions_clip, predictions_bin, eps_list, Y_test, adv_x_dict, output_csv_fpath):
    fieldnames = ['eps', 'accuracy_raw', 'accuracy_clip', 'accuracy_bin']
    to_csv = []

    for eps in eps_list:
        X_test_adv = adv_x_dict[eps]

        # Evaluate the accuracy of the MNIST model on adversarial examples
        accuracy_raw = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
        accuracy_clip = tf_model_eval(sess, x, y, predictions_clip, X_test_adv, Y_test)
        accuracy_bin = tf_model_eval(sess, x, y, predictions_bin, X_test_adv, Y_test)

        print ('Test accuracy on adversarial examples: raw %.4f, clip %.4f, bin %.4f (eps=%.1f): '% (accuracy_raw, accuracy_clip, accuracy_bin, eps))

        to_csv.append({'eps': eps, 
                       'accuracy_raw': accuracy_raw, 
                       'accuracy_clip': accuracy_clip, 
                       'accuracy_bin': accuracy_bin,
                       })

    write_to_csv(to_csv, output_csv_fpath, fieldnames)


def calculate_accuracy_adv_jsma(sess, x, y, predictions, Y_test, X_test, X_test_adv, output_csv_fpath):
    fieldnames = ['width', 'height', 'accuracy_legitimate', 'accuracy_malicious']
    to_csv = []

    print ("\n===Calculating the accuracy with feature squeezing...")
    for width in range(1, 11):
        # height = width
        for height in range(1, 11):
            X_squeezed = median_filter_np(X_test, width, height)
            X_adv_squeezed = median_filter_np(X_test_adv, width, height)

            accuracy_leg = tf_model_eval(sess, x, y, predictions, X_squeezed, Y_test)
            accuracy_mal = tf_model_eval(sess, x, y, predictions, X_adv_squeezed, Y_test)

            to_csv.append({'width': width, 'height': height, 'accuracy_legitimate': accuracy_leg, 'accuracy_malicious': accuracy_mal})
            print ("Width: %2d, Height: %2d, Accuracy_legitimate: %.2f, Accuracy_malicious: %.2f" % (width, height, accuracy_leg, accuracy_mal))

    write_to_csv(to_csv, output_csv_fpath, fieldnames)


def calculate_l1_distance_fgsm(sess, x, predictions_orig, predictions_squeezed, adv_x_dict, csv_fpath):
    print ("\n===Calculating L1 distance with feature squeezing...")
    eps_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    l1_dist = np.zeros((10000, 11))
    for i, eps in enumerate(eps_list):
        print("Epsilon=",eps)
        X_test_adv = adv_x_dict[eps]
        l1_dist_vec = tf_model_eval_distance(sess, x, predictions_orig, predictions_squeezed, X_test_adv)
        l1_dist[:,i] = l1_dist_vec

    np.savetxt(csv_fpath, l1_dist, delimiter=',')
    print ("---Results are stored in ", csv_fpath, '\n')
    return l1_dist


def calculate_l1_distance_jsma(sess, x, predictions, X_test, X_test_adv, csv_fpath):
    print ("\n===Calculating L1 distance with feature squeezing...")
    nb_examples = len(X_test_adv)

    l1_dist = np.zeros((nb_examples, 2))

    # for i, k_width in enumerate(range(1,11)):
    for i, k_width in [(0, 3)]:
        X_test_adv_smoothed = median_filter_np(X_test_adv, k_width)
        X_test_smoothed = median_filter_np(X_test, k_width)

        l1_dist_vec = tf_model_eval_distance_dual_input(sess, x, predictions, X_test, X_test_smoothed)
        l1_dist[:,2*i] = l1_dist_vec
        
        l1_dist_vec = tf_model_eval_distance_dual_input(sess, x, predictions, X_test_adv, X_test_adv_smoothed)
        l1_dist[:,2*i+1] = l1_dist_vec

    np.savetxt(csv_fpath, l1_dist, delimiter=',')
    print ("---Results are stored in ", csv_fpath, '\n')
    return l1_dist


def calculate_l1_distance_joint(sess, x, predictions, X_test, X_test_adv_fgsm, X_test_adv_jsma, csv_fpath):
    print ("\n===Calculating max(L1) distance with feature squeezing...")
    nb_examples = max(len(X_test), len(X_test_adv_fgsm), len(X_test_adv_jsma))

    l1_dist = np.zeros((nb_examples, 3))
    median_filter_width = 3

    for i, X in enumerate([X_test, X_test_adv_fgsm, X_test_adv_jsma]):
        X_test1 = X
        X_test2 = median_filter_np(X_test1, median_filter_width)
        X_test3 = binary_filter_np(X_test1)

        l1_dist_vec = tf_model_eval_dist_tri_input(sess, x, predictions, X_test1, X_test2, X_test3, mode = 'max')
        l1_dist[:len(X),i] = l1_dist_vec

    np.savetxt(csv_fpath, l1_dist, delimiter=',')
    print ("---Results are stored in ", csv_fpath, '\n')
    return l1_dist

def main(argv=None):
    sess = load_tf_session()
    print ("\n===Loading MNIST data...")
    X_train, Y_train, X_test, Y_test = get_mnist_data()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    with tf.variable_scope('mnist_original'):
        model = model_mnist()
        predictions = model(x)
        # predictions_bin = model(binary_filter_tf(x))
        predictions_bin = model(reduce_precision_tf(x, npp=2))
        predictions_clip = model(tf.clip_by_value(x, 0., 1.))
    print("\n===Defined TensorFlow model graph.")

    # Load an MNIST model
    maybe_download_mnist_model()
    model_name = 'mnist_epochs%d' % FLAGS.nb_epochs
    mnist_model_path = os.path.join(FLAGS.train_dir, model_name)
    original_variables = [k for k in tf.global_variables() if k.name.startswith('mnist_original')]
    tf_model_load_from_path(sess, mnist_model_path, original_variables)
    print ("---Loaded a pre-trained MNIST model.\n")

    accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    print('Test accuracy on raw legitimate examples ' + str(accuracy))

    result_folder = 'results/mnist'
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    if FLAGS.task == 'FGSM':
        nb_examples = 10000
        eps_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        adv_x_dict = get_fgsm_adv_examples(sess, x, predictions, X_test, eps_list, model_name, nb_examples, result_folder)

        if FLAGS.visualize is True:
            img_fpath = os.path.join(result_folder, model_name + '_FGSM_examples.png')
            draw_fgsm_adv_examples(adv_x_dict, Y_test, img_fpath)
            print ('\n===Adversarial images are saved in ', img_fpath)

        csv_fpath = model_name + "_fgsm_squeezing_accuracy_%dexamples.csv" % nb_examples
        csv_fpath = os.path.join(result_folder, csv_fpath)

        print ("\n===Calculating the accuracy with feature squeezing...")
        if not os.path.isfile(csv_fpath):
            calculate_accuracy_adv_fgsm(sess, x, y, predictions, predictions_clip, predictions_bin, eps_list, Y_test, adv_x_dict, csv_fpath)
        print ("---Results are stored in ", csv_fpath, '\n')

    elif FLAGS.task == 'FGSM-adv-train':
        # Load an adversarially trained MNIST model for comparison.
        with tf.variable_scope('mnist_adv_train'):
            model_2 = model_mnist()
            predictions_at = model_2(x)
            predictions_at_bin = model_2(reduce_precision_tf(x, npp=2))
            predictions_at_clip = model_2(tf.clip_by_value(x, 0., 1.))

        model_name = 'mnist_adv_train_epochs%d' % FLAGS.nb_epochs
        mnist_model_path = os.path.join(FLAGS.adv_train_dir, model_name)
        adv_train_variables = [k for k in tf.global_variables() if k.name.startswith('mnist_adv_train')]
        tf_model_load_from_path(sess, mnist_model_path, adv_train_variables)
        print ("---Loaded an adversarially pre-trained MNIST model.\n")

        accuracy = tf_model_eval(sess, x, y, predictions_at, X_test, Y_test)
        print('Test accuracy on raw legitimate examples (adv-trained-model) ' + str(accuracy))

        # Get adversarial examples.
        nb_examples = 10000
        eps_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        adv_x_dict = get_fgsm_adv_examples(sess, x, predictions_at, X_test, eps_list, model_name, nb_examples, result_folder)

        csv_fpath = model_name + "_fgsm_squeezing_accuracy_%dexamples.csv" % nb_examples
        csv_fpath = os.path.join(result_folder, csv_fpath)

        print ("\n===Calculating the accuracy with feature squeezing...")
        if not os.path.isfile(csv_fpath):
            calculate_accuracy_adv_fgsm(sess, x, y, predictions_at, predictions_at_clip, predictions_at_bin, eps_list, Y_test, adv_x_dict, csv_fpath)
        print ("---Results are stored in ", csv_fpath, '\n')

    elif FLAGS.task == 'JSMA':
        # Generate or load JSMA adversarial examples.
        nb_examples = 1000
        X_adv = get_jsma_adv_examples(sess, x, predictions, X_test, Y_test, model_name, nb_examples, result_folder)

        if FLAGS.visualize is True:
            img_fpath = os.path.join(result_folder, model_name + '_JSMA_examples.png')
            draw_jsma_adv_examples(X_adv, X_test, Y_test, img_fpath)
            print ('\n===Adversarial images are saved in ', img_fpath)

        csv_fpath = model_name + "_jsma_squeezing_accuracy_%dexamples.csv" % nb_examples
        csv_fpath = os.path.join(result_folder, csv_fpath)
        print ("\n===Calculating the accuracy with feature squeezing...")
        if not os.path.isfile(csv_fpath):
            calculate_accuracy_adv_jsma(sess, x, y, predictions, Y_test[:nb_examples], X_test[:nb_examples], X_adv, csv_fpath)
        print ("---Results are stored in ", csv_fpath, '\n')

    elif FLAGS.task == 'JSMA-detection':
        # Calculate L1 distance on prediction for JSMA adversarial detection.
        nb_examples = 1000
        X_adv = get_jsma_adv_examples(sess, x, predictions, X_test, Y_test, model_name, nb_examples, result_folder)

        csv_fpath = model_name + "_jsma_l1_distance_%dexamples.csv" % nb_examples
        csv_fpath = os.path.join(result_folder, csv_fpath)
        if not os.path.isfile(csv_fpath):
            l1_dist = calculate_l1_distance_jsma(sess, x, predictions_clip, X_test[:nb_examples], X_adv, csv_fpath)
        else:
            l1_dist = np.loadtxt(csv_fpath, delimiter=',')

        # Train a detector by selecting a threshold.
        print ("\n===Training an JSMA detector...")
        size_train = size_val = int(nb_examples/2)
        col_id_leg = [0]
        col_id_adv = [1]

        x_train = np.hstack( [ l1_dist[:size_train, col_id] for col_id in col_id_leg+col_id_adv ] )
        y_train = np.hstack([np.zeros(size_train*len(col_id_leg)), np.ones(size_train*len(col_id_adv))])

        x_val = np.hstack( [l1_dist[-size_val:, col_id] for col_id in col_id_leg+col_id_adv ])
        y_val = np.hstack([np.zeros(size_val*len(col_id_leg)), np.ones(size_val*len(col_id_adv))])

        train_detector(x_train, y_train, x_val, y_val)
        print ("---Done")

    elif FLAGS.task == 'FGSM-detection':
        nb_examples = 10000
        eps_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        adv_x_dict = get_fgsm_adv_examples(sess, x, predictions, X_test, eps_list, model_name, nb_examples, result_folder)

        csv_fpath = model_name + "_fgsm_squeezing_accuracy_%dexamples.csv" % nb_examples
        csv_fpath = os.path.join(result_folder, csv_fpath)

        # Calculate L1 distance on prediction for adversarial detection.
        csv_fpath = model_name + "_fgsm_l1_distance_%dexamples.csv" % nb_examples
        csv_fpath = os.path.join(result_folder, csv_fpath)
        if not os.path.isfile(csv_fpath):
            l1_dist = calculate_l1_distance_fgsm(sess, x, predictions_clip, predictions_bin, adv_x_dict, csv_fpath)
        else:
            l1_dist = np.loadtxt(csv_fpath, delimiter=',')

        # Train a detector by selecting a threshold.
        print ("\n===Training a FGSM detector...")
        size_train = size_val = int(nb_examples/2)
        col_id_leg = [0]
        # Selected epsilon: 0.1, 0.2, 0.3
        col_id_adv = [1,2,3]

        x_train = np.hstack( [ l1_dist[:size_train, col_id] for col_id in col_id_leg+col_id_adv ] )
        y_train = np.hstack([np.zeros(size_train*len(col_id_leg)), np.ones(size_train*len(col_id_adv))])

        x_val = np.hstack( [l1_dist[-size_val:, col_id] for col_id in col_id_leg+col_id_adv ])
        y_val = np.hstack([np.zeros(size_val*len(col_id_leg)), np.ones(size_val*len(col_id_adv))])

        train_detector(x_train, y_train, x_val, y_val)
        print ("---Done")

    elif FLAGS.task == 'joint-detection':
        nb_examples_jsma = 1000
        nb_examples_fgsm = 10000
        nb_examples_detection = min(nb_examples_jsma, nb_examples_fgsm)

        eps_list = [0.3]
        fgsm_adv_x_dict = get_fgsm_adv_examples(sess, x, predictions, X_test, eps_list, model_name, nb_examples_fgsm, result_folder)
        X_test_adv_jsma = get_jsma_adv_examples(sess, x, predictions, X_test, Y_test, model_name, nb_examples_jsma, result_folder)

        X_test_adv_fgsm = fgsm_adv_x_dict[0.3][:nb_examples_fgsm]
        X_test_adv_jsma = X_test_adv_jsma[:nb_examples_jsma]

        csv_fpath = model_name + "_joint_l1_distance_%dexamples.csv" % nb_examples_detection
        csv_fpath = os.path.join(result_folder, csv_fpath)
        if not os.path.isfile(csv_fpath):
            l1_dist = calculate_l1_distance_joint(sess, x, predictions_clip, X_test, X_test_adv_fgsm, X_test_adv_jsma, csv_fpath)
            np.savetxt(csv_fpath, l1_dist, delimiter=',')
            print ("---Results are stored in ", csv_fpath, '\n')
        else:
            l1_dist = np.loadtxt(csv_fpath, delimiter=',')
        
        # Train a detector by selecting a threshold.
        print ("\n===Training a joint detector...")
        nb_examples_min = min(len(X_test), len(X_test_adv_fgsm), len(X_test_adv_jsma))
        size_train = size_val = int(nb_examples_min/2)
        col_id_leg = [0]
        col_id_adv = [1,2]

        x_train = np.hstack( [ l1_dist[:size_train, col_id] for col_id in col_id_leg+col_id_adv ] )
        y_train = np.hstack([np.zeros(size_train*len(col_id_leg)), np.ones(size_train*len(col_id_adv))])

        x_val = np.hstack( [l1_dist[size_train:size_train+size_val, col_id] for col_id in col_id_leg+col_id_adv ])
        y_val = np.hstack([np.zeros(size_val*len(col_id_leg)), np.ones(size_val*len(col_id_adv))])

        train_detector(x_train, y_train, x_val, y_val)
        print ("---Done")

    else:
        print ("Please specify a task: FGSM, JSMA, FGSM-detection, JSMA-detection, joint-detection.")


if __name__ == '__main__':
    app.run()
