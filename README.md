# Feature Squeezing
*Detecting Adversarial Examples in Deep Neural Networks*  
  
<img src="http://evademl.org/images/squeezing.png" height="150" width="360" alt="idea_figure" >  

## Latest Updates

This repository is **now obsolete**.

**Please move on to our [EvadeML-Zoo](http://evademl.org/zoo/) project if you would like to conduct experiments on more datasets and models.**

The old repository is preserved here, but please use https://github.com/mzweilin/EvadeML-Zoo instead of this one.

## Run the code

### 1. Install dependencies.

```bash
pip install tensorflow==0.12.1 keras==1.2.0 pillow scikit-learn
```

If you are going to run experiment on GPU, you should install `tensorflow-gpu` instead of `tensorflow`.

Cleverhans v1.0.0 will be automatically fetched and located when executing `from utils import load_externals` in Python. You don't need to do anything on it.

### 2. Run the experiments on MNIST.

(Optional) Train a classification model on MNIST, then use adversarial training to get a second model.
```bash
python train_mnist_model.py
```

If you don't train your own models, the program `python mnist_experiment.py` will automatically download the pre-trained ones from our website.

First, let's test the color bit depth reduction with FGSM.
```bash
python mnist_experiment.py --task FGSM --visualize
```
The program will generate adversarial examples with FGSM, output a figure with image examples, and evaluate the model accuracy with and without the binary filter. In order to save time, the program only generates adversarial examples in the first run, and the later runs will reuse the adversarial examples.

Second, we will test adversarial training with and without the binary filter, so as to compare adversarial training with feature squeezing.
```bash
python mnist_experiment.py --task FGSM-adv-train
```

Next, we will test the median smoothing with JSMA.
```bash
python mnist_experiment.py --task JSMA --visualize
```
The program will generate adversarial examples with JSMA, output a figure with image examples, and evaluate the model accuracy with and without median smoothing.

Finally, we will conduct three detection experiments. The program will report the detection performance as well as the selected thresholds.


<img src="http://evademl.org/images/squeezingframework.png" height="120" width="360" alt="idea_figure" >  

```bash
python mnist_experiment.py --task FGSM-detection
python mnist_experiment.py --task JSMA-detection
python mnist_experiment.py --task joint-detection
```

### 3. Review the results.

The experimental results are stored in `./results/mnist/`, including tables, figures, and pickled files.

You can also download the results we generated before using one GeForce GTX 1080.

```bash
mkdir results && cd results
wget http://www.cs.virginia.edu/~wx4ed/downloads/squeezing/results_mnist.tar.gz
tar xfz results_mnist.tar.gz
rm results_mnist.tar.gz
cd ..
python mnist_experiment.py --task joint-detection
```

## Cite this work

You are encouraged to cite the following paper if you use `Feature Squeezing` for academic research.

```
@article{xu2017feature,
  title={Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks},
  author={Xu, Weilin and Evans, David and Qi, Yanjun},
  journal={arXiv preprint arXiv:1704.01155},
  year={2017}
}
```
