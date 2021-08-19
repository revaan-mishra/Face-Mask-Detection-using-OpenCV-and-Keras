
<h1 align="center">Face Mask Detection using Keras and Tensorflow</h1>

<div align="center">
    <strong>A lightweight face mask detection that is easy to deploy</strong>
</div>

<div align="center">
    Trained on Tensorflow/Keras. 
</div>

<br/>
<br/>

## Table of Contents
- [Features](#features)
- [About](#about)
- [Frameworks and Libraries](#frameworkslibraries)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Setup](#setup) 
- [How to Run](#how-to-run)
- [Credits](#credits)

## Features
- __Lightweight models:__  only `2,422,339` and `2,422,210` parameters for the MFN and RMFD models, respectively
- __Detection of multiple faces:__ able to detect multiple faces in one frame
- __Support for detection in webcam stream:__ our app supports detection in images and video streams 
- __Support for detection of improper mask wearing:__ our MFN model is able to detect improper mask wearing including
  (1) uncovered chin, (2) uncovered nose, and (3) uncovered nose and mouth.

## About
This app detects human faces and proper mask wearing in images and webcam streams. 

Under the COVID-19 pandemic, wearing
mask has shown to be an effective means to control the spread of virus. The demand for an effective mask detection on 
embedded systems of limited computing capabilities has surged, especially in highly populated areas such as public 
transportations, hospitals, etc. Trained on MobileNetV2, a state-of-the-art lightweight deep learning model on 
image classification, the app is computationally efficient to deploy to help control the spread of the disease.

While many work on face mask detection has been developed since the start of the pandemic, few distinguishes whether a
mask is worn correctly or incorrectly. Given the discovery of the new coronavirus variant in UK, we aim to provide a 
more precise detection model to help strengthen enforcement of mask mandate around the world.

## Frameworks and Libraries
- __[OpenCV](https://opencv.org/):__ computer vision library used to process images
- __[OpenCV DNN Face Detector](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py):__ 
  Caffe-based Single Shot-Multibox Detector (SSD) model used to detect faces
- __[Tensorflow](https://www.tensorflow.org/) / [Keras](https://keras.io/):__ deep learning framework used to build and train our models
- __[MobileNet V2](https://arxiv.org/abs/1801.04381):__ lightweight pre-trained model available in Keras Applications; 
  used as a base model for our transfer learning

## Datasets
We provide two models trained on two different datasets. 
Our RMFD dataset is built from the [Real World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) and 
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset).

### RMFD dataset
This dataset consists of __3,833__ images:
- `face_no_mask`: 1,915 images
- `face_with_mask`: 1,918 images

Each image is a cropped real-world face image of unfixed sizes. The `face_no_mask` data is randomly sampled from the 90,568 no mask
data from the Real World Masked Face Dataset and the `face_with_mask` data entirely provided by the original dataset.

The `face_with_mask_correctly` and `face_with_mask_incorrectly` classes consist of the resized 128*128 images from 
the original MaskedFace-Net work without any sampling. The `face_no_mask` is built from the 
Flickr-Faces-HQ Dataset (FFHQ) upon which the MaskedFace-Net data was created.
All images in MaskedFace-Net are morphed mask-wearing images and `face_with_mask_incorrectly` consists of 10% uncovered chin, 10% uncovered nose, and 80% uncovered nose and mouth images.

### Download
The dataset is now available [here](https://drive.google.com/file/d/1Y1Y67osv8UBKn_ANckCXPvY2aZqv1Cha/view?usp=sharing)! (July 01, 2021)

## Requirements
This project is built using Python 3.7 on Windows 10. The training of the model is performed on custom GCP 
Compute Engine (8 vCPUs, 15.75 GB memory) with `tensorflow==2.4.0`. All dependencies and packages are listed in
`requirements.txt`. 

Note: We used `opencv-python-headless==4.5.1` due to an [issue](https://github.com/skvark/opencv-python/issues/423).
However, recent release of `opencv-python 4.5.1.48` seems to have fixed the problem.
Feel free to modify the `requirements.txt` before you install all the listed packages.

## Setup
1. Open your terminal, `cd` into where you'd like to clone this project, and clone the project:
```
$ git clone https://github.com/revaan-mishra/Face-Mask-Detection-using-OpenCV-and-Keras-master.git
```
2. Download and install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html).
3. Create an environment with the packages on `requirements.txt` installed:
```
$ conda create --name env_name --file requirements.txt
```
4. Now you can `cd` into the clone repository to run or inspect the code.

## How to Run

### To detect masked faces in images
`cd` into `/src/` and enter the following command:
```
$ python detect_mask_images.py -i <image-path> [-m <model>] [-c <confidence>]
```

### To detect masked faces in webcam streams
`cd` into `/src/` and enter the following command:
```
$ python detect_mask_video.py [-m <model>] [-c <confidence>]
```

### To train the model again on the dataset
`cd` into `/src/` and enter the following command:
```
$ python train.py [-d <dataset>]
```
Make sure to modify the paths in `train.py` to avoid overwriting existing models.

Note: 
- `<image-path>` should be relative to the project root directory instead of `/src/`
- `<model>` should be of `str` type; accepted values are `MFN` and `RMFD` with default value `MFN`
- `<confidence>` should be `float`; accepting values between `0` and `1` with default value `0.5`
- `<dataset>` should be of `str` type; accepted values are `RMFD`

## Credits
- [Real-World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) ，RMFD）
- Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of 
  correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, 
  Elsevier, 2020, [DOI:10.1016/j.smhl.2020.100144](https://doi.org/10.1016/j.smhl.2020.100144)
- Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection 
  mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", 
  CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 
  2020, [DOI:10.32604/cmes.2020.011663](DOI:10.32604/cmes.2020.011663)
- [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
- [Face Mask Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- [Object Detection](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-object-detection)
- Co-authored by [Rudraksh Bharadwaj](https://github.com/rudrakshbhardwaj) and [Samarth Sajawan](https://github.com/SamarthSajwan) 
