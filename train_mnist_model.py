from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import keras

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils import load_externals
from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval
from cleverhans.attacks import fgsm

from utils.utils_tf import tf_model_train_and_save

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', './trained_models/mnist', 'Directory storing the saved model.')
flags.DEFINE_string('adv_train_dir', './trained_models/mnist_adv_train', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 100, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def load_tf_session():
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")
    return sess


# Get MNIST test data
def get_mnist_data():
    X_train, Y_train, X_test, Y_test = data_mnist()
    print("Loaded MNIST test data.")

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    return X_train, Y_train, X_test, Y_test


def train_mnist(sess, X_train, Y_train, X_test, Y_test, save_path):
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # Define TF model graph
    with tf.variable_scope('mnist_original'):
        model = model_mnist()
        predictions = model(x)
        
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Train and save an MNIST model.
    tf_model_train_and_save(sess, x, y, predictions, X_train, Y_train, save_path=save_path , evaluate=evaluate)


def adv_train_mnist(sess, X_train, Y_train, X_test, Y_test, save_path):
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Redefine TF model graph
    with tf.variable_scope('mnist_adv_train'):
        model_2 = model_mnist()
        predictions_2 = model_2(x)
        adv_x_2 = fgsm(x, predictions_2, eps=0.3, clip_min=0., clip_max=1.)
        predictions_2_adv = model_2(adv_x_2)

    def evaluate_2():
        # Evaluate the accuracy of the adversarialy trained MNIST model on
        # legitimate test examples
        accuracy = tf_model_eval(sess, x, y, predictions_2, X_test, Y_test)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        accuracy_adv = tf_model_eval(sess, x, y, predictions_2_adv, X_test, Y_test)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

    # Perform adversarial training
    tf_model_train_and_save(sess, x, y, predictions_2, X_train, Y_train, save_path=save_path, predictions_adv=predictions_2_adv,
        evaluate=evaluate_2)


def main(argv=None):
    """
    Train MNIST models
    :return:
    """
    model_name = 'mnist_epochs%d' % FLAGS.nb_epochs
    mnist_model_path = os.path.join(FLAGS.train_dir, model_name)

    adv_train_model_name = 'mnist_adv_train_epochs%d' % FLAGS.nb_epochs
    adv_trained_mnist_model_path = os.path.join(FLAGS.adv_train_dir, adv_train_model_name)

    for dir_name in [FLAGS.train_dir, FLAGS.adv_train_dir]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    sess = load_tf_session()
    X_train, Y_train, X_test, Y_test = get_mnist_data()

    train_mnist(sess, X_train, Y_train, X_test, Y_test, mnist_model_path)
    adv_train_mnist(sess, X_train, Y_train, X_test, Y_test, adv_trained_mnist_model_path)


if __name__ == '__main__':
    app.run()
