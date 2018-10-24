# SmartBox



## Introduction

This toolbox is a collection of various attack, detection and mitigation algorithms which can be used by researchers to carry out experiments. Along with providing the algorithms in a single user-friendly framework, the toolbox also provides some useful operations specifically designed to analyse experiment results easily.

### Prerequisites

Following are the libraries that need to be installed before you use the toolbox:

```
numpy
keras
tensorflow
cv2
sklearn
```

Apart from these libraries, the toolbox requires a CNN model with the following class attributes:

1. **num_labels :** Number of labels in the dataset.
2. **get_logits(x) :** Returns the output of the logits layer corresponding to the symbolic input x.


## Documentation

The toolbox is broadly divided in three major modules

```
Attack
Detection
Mitigation
```

### Attack

This module consists of implementation of some of the major attack generation algorithms available:

1. **FGSM**
2. **PGD**
3. **MI-FGSM**
4. **C&W L2**
5. **EAD (with L1 rule)**
6. **EAD (with EN rule)**
7. **Deepfool**

Each of these attack implementations inherit Attack class from Attack.py in Attack module and override the setup() and attack() method. Any new attack must also do the same.

Apart from the methods particular to a certain attack, following methods are common for all:

1. **setup(x,y) :** Creates a graph for attacking images using the input and output placeholders x and y.
2. **attack(X_test, Y_test) :** Runs the attack algorithm on the supplied set of images X_test.
3. **analyse(X_test, Y_test, save) :** Creates and returns a dictionary of classification accuracies of the computed adversaries and X_test corresponding to the true labels Y_test. If the save attribute is not None then, original and adversary images are saved for comparison in the provided directory. 
4. **print_analysis_results(X_test, Y_test, save) :** Prints the analysis dictionary created by analyse function.
5. **get_perturbation() :** Returns the perturbations added to original images to convert them into adversaries.


Sample code :

```python
from Attacks.fgsm import FGSM

a = FGSM(sess, model)
adv = a.attack(X_test, Y_test)
a.print_analysis_results(X_test, Y_test, save="./")
```

### Detection

This module consists of implementation of some of the major detection algorithms available:

1. **Adaptive Noise Reduction**
2. **PCA Detect**
3. **Convolution Filter Detection**
4. **Artifact Learning**


Each of these detection implementations inherit Detection class from Detection.py in Detection module and override the detect() method. Any new detection algorithm must also do the same.

Apart from the methods particular to a certain algorithm, following methods are common for all:

1. **detect(X) :** Runs the detection algorithm on the supplied set of images X.
2. **return_detected() :** Returns the images detected as adversaries in the detect(X) method.
3. **prepare_data(X_test,adv) :** Shuffles the two supplied arguments together and along with the shuffled combined set of images it returns an array of true labels i.e. an array indicating whether an image in a particular index is an adversary or not.
4. **analyse(Y_test) :** Creates and returns a dictionary of various evaluation metrics that test the effectiveness of the performed detection procedure.
5. **print_analysis_results(Y_test) :** Prints the analysis dictionary created by analyse function.


Sample code :

```python
from Detection.LearnArtifacts import LearnArtifactsModel

a=LearnArtifactsModel(sess, model, height, width, attack, trainX, trainY)
x, y = a.prepare_data(X_test, adv)
det = a.detect(x)
a.print_analysis_results(y)
```


### Mitigation

This module consists of implementation of some of the major mitigation algorithms available:

1. **Adaptive Noise Reduction**
2. **Denoising AutoEncoder**
3. **Adversarial Training**
4. **Gaussian Blur**
5. **Randomization**


Each of these mitigation implementations inherit Mitigation class from Mitigation.py in Mitigation module and override the mitigate() method. Any new mitigation algorithm must also do the same.

Apart from the methods particular to a certain algorithm, following methods are common for all:

1. **mitigate(X) :** Runs the mitigation algorithm on the supplied set of images X.
2. **return_mitigated() :** Returns the images after running the mitigation algorithm.
3. **analyse(X_test,Y_test,save) :** Creates and returns a dictionary of classification acuracy before and after mitigation. If the save attribute is not None then, images before and after mitigation are saved for comparison in the supplied directory. 
4. **print_analysis_results(X_test,Y_test,save) :** Prints the analysis dictionary created by analyse function.


Sample code :

```python
from Mitigation.GaussianBlur import GaussianBlur

a = GaussianBlur(sess, model, height, width)
mit = a.mitigate(X_test)
a.print_analysis_results(X_test, Y_test, save="./")
```

## Usage

The toolbox can be operated from the command line by supplying appropriate arguments. 

Example : 

```
python smartbox.py --new_dataset --dataset_location <DATASET LOCATION> --save_dataset_info <FILE LOCATION> --load_model <FILE LOCATION> --attack L2 --detect_algo ArtifactLearning --mitigate_algo GaussianBlur --identification --verification
```
