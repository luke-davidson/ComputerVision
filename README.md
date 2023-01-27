# Computer Vision
This repo holds all programming assignments completed for my Computer Vision course (Fall 2021).

***Note:*** Scaffolding code was given for some of these assignments. All code beneath functions and methods is completed by me, unless otherwise noted.

# Assignment Descriptions

## PA1 --- NCC Image Layering
Layering of red, green and blue image filters based on a normalized cross correlation (NCC) calculation of image pyramids.

- **Code:** [`pa1_NCC_Image_Layering/pa1.ipynb`](https://github.com/luke-davidson/ComputerVision/blob/main/pa1_NCC_Image_Layering/pa1.ipynb)

## PA2 --- SIFT Feature Extraction + Matching
Implementation of SIFT feature extraction and corner matching: sobel kernels, second moment matrices, max pooling, patch descriptors, pairwise feature distances, gradient magnitudes and orientations, feature matching.

- **Code:** [`pa2_SIFT_Feature_Matching/pa2_code/pa2.ipynb`](https://github.com/luke-davidson/ComputerVision/blob/main/pa2_SIFT_Feature_Matching/pa2_code/pa2.ipynb)
- **Report:** [`pa2_SIFT_Feature_Matching/pa2_report.pdf`](https://github.com/luke-davidson/ComputerVision/blob/main/pa2_SIFT_Feature_Matching/pa2_report.pdf)

## PA3 --- Camera Calibration + Fundamental Matrix Estimation
Estimating camera projection and fundamental matrices using RANSAC to match images at different viewpoints.

- **Code:** [`pa3_CameraCal_RANSAC_Matrix_Estimations/pa3_code/pa3.ipynb`](https://github.com/luke-davidson/ComputerVision/blob/main/pa3_CameraCal_RANSAC_Matrix_Estimations/pa3_code/pa3.ipynb)
- **Report:** [`pa3_CameraCal_RANSAC_Matrix_Estimations/pa3_report.pdf`](https://github.com/luke-davidson/ComputerVision/blob/main/pa3_CameraCal_RANSAC_Matrix_Estimations/pa3_report.pdf)

## PA4 --- Fully Connected Neural Network (FCN)
Linear Classifier: Implementation of a linear classifier using cross entropy and SVM Hinge losses.
FC Net: Implementation of a fully connected neural network. Processes implemented include forward and backward passes of affine layers, ReLU activation, softmax loss, SGD and momentum.

- **Code:** 
	- [`pa4_Fully_Connected_Net/pa4_code/linear_classifier.ipynb`](https://github.com/luke-davidson/ComputerVision/blob/main/pa4_Fully_Connected_Net/pa4_code/linear_classifier.ipynb)
	- [`pa4_Fully_Connected_Net/pa4_code/fc_net.ipynb`](https://github.com/luke-davidson/ComputerVision/blob/main/pa4_Fully_Connected_Net/pa4_code/fc_net.ipynb)

## PA5 --- Convolutional Neural Network (CNN)
Full naive implementation of a CNN. Naive implementations of forward and backward passes of convolution layers, batch normalization (normal and spatial) and adaptive average pooling. Network trained using cross-validation.

- **Code:** [`pa5_CNN/pa5_1_code/cnn.ipynb`](https://github.com/luke-davidson/ComputerVision/blob/main/pa5_CNN/pa5_1_code/cnn.ipynb)
