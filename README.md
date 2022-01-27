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



# Description
With the ever-increasing survey speed of optical wide-field telescopes, rapid and reliable source localization is paramount. We propose a new way to analyze optical imaging
data that uses computer vision techniques that can naturally deal with large amounts of data. We present AutoSourceID-Light (ASID-L), an innovative framework for rapidly localizing sources in optical images.

## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)


<!-- TABLE OF CONTENTS -->
<!-- 
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
--> 




# Installation
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white&style=plastic"/> ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white&style=plastic)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white&style=plastic)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white&style=plastic)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white&style=plastic)


_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/FiorenSt/AutoSourceID-Light.git
   ```
2. Download Zenodo folder for training/test/validation sets
   ```sh
   curl https://zenodo.org/...
   ```
3. Save the files in the folder Training Set

# Dependencies:

* Python 3 (or superior)
* TensorFlow 2.0
  * Keras
* Scikit-Image Version?
* Numpy Verions?
* Joblib Version?
* Patchify
* OpenCV
* Astropy



<<<<<<< HEAD
## Step-by-step
Training set: 
 * Load image
 * Normalize
 * 256x256 pathces
=======
# Usage
Here we introduce a simplified version of all the steps of ASID-L, for more detailed information check out the ASID-L paper.





Training Set

The training, test and validation set are images from the MeerLICHT telescope while the locations are obtained from GAIA EDR3.
Although important, the user can skip this step being the U-Net already trained for a series of different SNRs thresholds.
The training set is made of 3 10496x10496 optical fields divided in 5043 patches of 256x256. Appriximately 80% training, 10% test and 10% validation.
A normalization step is applied to each field separately.
A U_net also needs the mask training, test and validation set. This is made stargting from GAIA EDR3 locations and then patchified in the same way as the optical images.

>>>>>>> c8e7fd4339a668db5f904a5d4b65344959edce04

U-Net
 * Load mode
 * OR
 * Specific U-Net structure
 
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


