
<!--<img src=https://see.fontimg.com/api/renderfont4/qZWEx/eyJyIjoiZnMiLCJoIjoxMDAsInciOjEwMDAsImZzIjoxMDAsImZnYyI6IiMwRjlCRkEiLCJiZ2MiOiIjMEMwMDAwIiwidCI6MX0/QXV0b1NvdXJjZUlELUxpZ2h0/beuna-line-regular.png>
-->

<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjoxMDAsInciOjEwMDAsImZzIjoxMDAsImZnYyI6IiMwRjlCRkEiLCJiZ2MiOiIjMEMwMDAwIiwidCI6MX0/QXV0b1NvdXJjZUlELUxpZ2h0/kg-second-chances-sketch.png>




<!--<img src=https://github.com/FiorenSt/AutoSourceID-Light/blob/main/ASID.PNG width=25% height=25%> <img src=https://github.com/FiorenSt/AutoSourceID-Light/blob/main/ASID-L.PNG width=25.5% height=25%> 
-->

<!--<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/PredictedMaskPatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGPatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=25% height=25%> -->

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=50% height=50%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=50% height=50%> 





![GitHub repo size](https://img.shields.io/github/repo-size/FiorenSt/AutoSourceID-Light?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/FiorenSt/AutoSourceID-Light?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/FiorenSt/AutoSourceID-Light?color=red&style=plastic)

<!--
![GitHub stars](https://img.shields.io/github/stars/FiorenSt/AutoSourceID-Light?style=social)
![GitHub forks](https://img.shields.io/github/forks/FiorenSt/AutoSourceID-Light?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/FiorenSt/AutoSourceID-Light?style=social)
![GitHub followers](https://img.shields.io/github/followers/FiorenSt?style=social)
-->



## Description
With the ever-increasing survey speed of optical wide-field telescopes, rapid and reliable source localization is paramount. We propose a new way to analyze optical imaging
data that uses computer vision techniques that can naturally deal with large amounts of data. We present AutoSourceID-Light (ASID-L), an innovative framework for rapidly localizing sources in optical images.

## Table of Contents 
- [Installation](#installation)
- [Step-by-step](#step-by-step)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white&style=plastic"/> ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white&style=plastic)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white&style=plastic)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white&style=plastic)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white&style=plastic)


Dependencies:

* Python 3 (or superior)
* TensorFlow 2.0
  * Keras
* Scikit-Image
* Numpy


## Step-by-step
Here we introduce a simplified version of all the steps of ASID-L, for more detailed information check out the ASID-L paper.

Training Set

The training, test and validation set are images from the MeerLICHT telescope while the locations are obtained from GAIA EDR3.
Although important, the user can skip this step being the U-Net already trained for a series of different SNRs thresholds.
The training set is made of 3 10496x10496 optical fields divided in 5043 patches of 256x256. Appriximately 80% training, 10% test and 10% validation.
A normalization step is applied to each field separately.
A U_net also needs the mask training, test and validation set. This is made stargting from GAIA EDR3 locations and then patchified in the same way as the optical images.


U-Net

The User can decide to either load one of the pre-trained models at different SNRs, or run the U-Net from scratch with new parameters choices:
 * Load model
 ```
 python LOAD_UNET.py "filename"
 ```
 * Specific U-Net structure
 Run the below command to moderate changes, as the number of epochs and the cut od SNR for the training set. 
 
 ```
 python RUN_UNET_fromScratch.py 'snr_threshold' 'epochs'
 ```
 Modify the file U_Net.py for major changes in the U-Net structure, as the number of layers etc.
 
 
 * Predict on 3 test images
 Benchmark the results on 3 different images of the test sets chosen by their sources density.
 
 
 
 Laplacian of Gaussian
 * Threshold and sigma parameters 
 
Output
 * Catalog of sources


## Usage
<!--
Provide instructions and examples for use. Include screenshots as needed.
To add a screenshot, create an `assets/images` folder in your repository and upload your screenshot to it. Then, using the relative filepath, add it to your README using the following syntax:
    ```md
    ![alt text](assets/images/screenshot.png)
    ```
-->

## Credits
<!--
List your collaborators, if any, with links to their GitHub profiles.
If you used any third-party assets that require attribution, list the creators with links to their primary web presence in this section.
If you followed tutorials, include links to those here as well.
-->

## License
<!--
The last section of a high-quality README file is the license. This lets other developers know what they can and cannot do with your project. If you need help choosing a license, refer to [https://choosealicense.com/](https://choosealicense.com/).
-->


## Features
<!--
If your project has a lot of features, list them here.
-->

