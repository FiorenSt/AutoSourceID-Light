
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
- [License](#license)
- [Features](#features)


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

3. Save the files in a folder "TrainingSet" and include the folder in the ASID-L repository

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
The use of the pre-trained ASID-L is straight forward: 
* Load the image and the pre-trained model
 ```
 python ASID-L.py 'DATA_PATH' 'MODEL_PATH'
 ```
 * A catalog 'coordinates.txt' of the localized sources will be created in the folder RESULTS.


For more detailed information on the trained model check out the ASID-L paper.


### Train U-Net from scratch

* Specify parameters of your choice, such as the number of epochs, the layers and the SNR cut-off modifying the file 'U_Net.py' 

* To train the U-Net without additional changes run:
 ```
 python ASID-L.py 'DATA_PATH' 'MODEL_PATH' 'load_model=False'
 ```
 
# License
Copyright 2022 Fiorenzo Stoppa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.




# Features

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/HSTFieldM16.png " >
<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/HSTField10396.png " >

