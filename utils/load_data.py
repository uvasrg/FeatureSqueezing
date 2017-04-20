import tarfile
from urllib2 import urlopen
import os
import pickle
import time

trained_mnist_model_url = "http://www.cs.virginia.edu/~wx4ed/downloads/squeezing/mnist_trained_model.tar.gz"
adv_trained_mnist_model_url = "http://www.cs.virginia.edu/~wx4ed/downloads/squeezing/mnist_clipped_adv_trained_model.tar.gz"


def download_and_extract_model(url, tgt_folder):
    tmp_fpath = "/tmp/" + os.path.basename(url)
    with open(tmp_fpath, "wb") as tmp_file:
        tmp_file.write(urlopen(url).read())
    with tarfile.open(tmp_fpath, 'r:gz') as tfile:
        tfile.extractall(tgt_folder)


def maybe_download_mnist_model():
    tgt_folder = './trained_models'
    if not os.path.isdir('./trained_models/mnist'):
        url = trained_mnist_model_url
        print "Downloading the pre-trained MNIST model from " + url
        download_and_extract_model(url, tgt_folder)

    if not os.path.isdir('./trained_models/mnist_adv_train'):
        url = adv_trained_mnist_model_url
        print "Downloading the adversarially pre-trained MNIST model from " + url
        download_and_extract_model(url, tgt_folder)


def restore_or_calculate_object(fpath, func, args, obj_name):
    if not os.path.isfile(fpath):
        print ("===Calculating %s..." % obj_name)
        start_time = time.time()
        obj = func(*args)
        duration = time.time() - start_time
        print ("Duration: %d sec" % duration)
        pickle.dump(obj, open(fpath, 'wb'))
    else:
        obj = pickle.load(open(fpath))
        print ("===Loaded %s." % obj_name)
    return obj
