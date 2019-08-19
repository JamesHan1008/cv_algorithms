# Computer Vision Algorithms

This project demonstrates object classification and detection algorithms commonly used in computer vision.


## Algorithms
- Object Classification
    - VGG-16
    - ResNet
- Object Detection
    - Viola-Jones
    - TensorFlow Object Detection API
        - SSD

### Algorithm Descriptions

#### VGG-16
VGG-16 is a pre-trained convolutional neural network with 16 layers. The first 15 layers acts as a feature transformer,
and the last layer uses these features to classify objects. Only the last layer needs to be trained on new labeled data
to be able to classify any arbitrary set of objects, and the weights on the first 15 layers can be reused.

#### ResNet
Similar to VGG-16, ResNet can also act as a feature transformer and its weights can be applied to a new data set using
transfer learning. ResNet is a convolutional neural network with a main branch and a shortcut branch, capable of not
only learning the main features but also learning from the residuals (what's not captured by the main features).

#### Viola-Jones
Viola-Jones is an older object detection algorithm mainly used for detecting faces and facial features. It depends on
pre-trained cascades to detect manually defined features and is not capable of training on new data.

#### SSD
SSD (single-shot multi-box detector) is a state-of-the-art object detection algorithm with advantages in detection speed
over FasterRCNN, but performs slightly worse in mAP (mean average precision). It divides an image into cells, and then
uses boxes of different sizes and aspect ratios within each cell to detect objects.


## Model Training
Not all models can be trained directly within this project, but for the ones that can, this is how to run it locally:
- `$ python3 -m algorithms.<algorithm_name>`
    - `-t`: train the model
    - `-l`: load existing model and continue training
    - `-e`: number of training epochs
    - `-d`: debugging mode
Currently the following models can be trained directly within this project:
- ResNet
- VGG-16

Models based on the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
can be trained within the TensorFlow package. To use a new trained model in this project, simply copy over the following
files to the appropriate algorithm resources directory:
- `frozen_inference_graph.pb`
- `label_map.pbtxt`


## Run Locally

### Set Up

#### Git LFS
This repository uses Git Large File Storage to store images, models, and other large files. After installing Git LFS,
run the following command:
- `git lfs pull`

#### Download Images
The images used in this project are not stored in this GitHub repository. Download them from the internet (instructions
are within each algorithm module) and place them in a folder named "images" in the root directory.

### Video Capture
Classifying and detecting objects captured in videos through the webcam.
- `$ python3 -m video_capture`
    - `-d`: object detection
    - `-c`: object classification
