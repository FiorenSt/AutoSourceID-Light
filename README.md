# AutoSourceID-Light

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/PredictedMaskPatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGPatch.png " width=25% height=25%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=25% height=25%>



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
Training set: 
 * Load image
 * Normalize
 * 256x256 pathces

U-Net
 * Load mode
 * OR
 * Specific U-Net structure
 
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

