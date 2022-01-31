
<!--<img src=https://see.fontimg.com/api/renderfont4/qZWEx/eyJyIjoiZnMiLCJoIjoxMDAsInciOjEwMDAsImZzIjoxMDAsImZnYyI6IiMwRjlCRkEiLCJiZ2MiOiIjMEMwMDAwIiwidCI6MX0/QXV0b1NvdXJjZUlELUxpZ2h0/beuna-line-regular.png>
-->

<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjoxMDAsInciOjEwMDAsImZzIjoxMDAsImZnYyI6IiMwRjlCRkEiLCJiZ2MiOiIjMEMwMDAwIiwidCI6MX0/QXV0b1NvdXJjZUlELUxpZ2h0/kg-second-chances-sketch.png>




<!--<img src=https://github.com/FiorenSt/AutoSourceID-Light/blob/main/ASID.PNG width=25% height=25%> <img src=https://github.com/FiorenSt/AutoSourceID-Light/blob/main/ASID-L.PNG width=25.5% height=25%> 
-->

<!--<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/PredictedMaskPatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGPatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=25% height=25%> -->

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=50% height=50%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=50% height=50%> 


<!--
![GitHub stars](https://img.shields.io/github/stars/FiorenSt/AutoSourceID-Light?style=social)
![GitHub forks](https://img.shields.io/github/forks/FiorenSt/AutoSourceID-Light?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/FiorenSt/AutoSourceID-Light?style=social)
![GitHub followers](https://img.shields.io/github/followers/FiorenSt?style=social)
-->



# Description
With the ever-increasing survey speed of optical wide-field telescopes, rapid and reliable source localization is paramount. We propose a new way to analyze optical imaging
data that uses computer vision techniques that can naturally deal with large amounts of data. We present AutoSourceID-Light (ASID-L), an innovative framework for rapidly localizing sources in optical images.

## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)


# Installation


_Follow the instructions below to download and start using ASID-L._

1. Clone the repo
   ```sh
   git clone https://github.com/FiorenSt/AutoSourceID-Light.git
   ```
2. Download Zenodo folder for training/test/validation sets 
   ```sh
   curl https://doi.org/10.5281/zenodo.5902893
   ```
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5902893.svg)](https://doi.org/10.5281/zenodo.5902893)

3. Save the files in a folder "Training Set" and include the folder in the ASID-L repository

# Dependencies:
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white&style=plastic"/> ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white&style=plastic)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white&style=plastic)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white&style=plastic)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white&style=plastic)

* Python 3 (or superior)
* TensorFlow 2.5
* Scikit-Image 0.18.1
* Numpy 1.20.3
* Joblib 1.0.1
* Patchify 0.2.3
* OpenCV 4.5.1
* Astropy 4.2.1



# Usage
Here we introduce a simplified version of all the steps of ASID-L, for more detailed information check out the ASID-L paper.





Training Set

The training, test and validation set are images from the MeerLICHT telescope while the locations are obtained from GAIA EDR3.
The training set is made of 3 10496x10496 field images divided in 5043 patches of 256x256 3 masks fields also patchified in the same way.

U-Net

You can decide to either load the pre-trained model at SNR=3, or run the U-Net from scratch with new parameters choices:
 
 * Load model and predict the locations
 ```
 python ASID-L.py 'DATA_PATH' 'MODEL_PATH'
 ```
 
 * Specific U-Net structure
 You can also train your own U-Net with specific paramters of your choice, such as the number of epochs and the SNR cut-off for the training set. 
 
 ```
 python ASID-L.py 'snr_threshold' 'epochs'
 ```
 Modify the file U_Net.py for major changes in the U-Net structure, as the number of layers etc.
 
 
 * Predict on 3 test images
 Benchmark the results on 3 different images of the test sets chosen by their sources density.
 
 
 
 Laplacian of Gaussian
 * Threshold and sigma parameters 
 
Output
 * Catalog of sources


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

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/HSTFieldM16.png " >
<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/HSTField10396.png " >

