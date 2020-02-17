# Audio captioning DCASE baseline system - 2020

Welcome to the repository of the audio captioning baseline system
for the DCASE challenge of 2020. 

Here you can find the complete code of the baseline system, consisting
of:  

  1. the caption evaluation part,
  2. the dataset pre-processing/feature extraction part,
  3. the data handling part for Pytorch library, and
  4. the deep neural network (DNN) method part
  
Parts 1, 2, and 3, also exist in separate repositories. Caption evaluation
tools for audio captioning can also be found
[here](https://github.com/audio-captioning/caption-evaluation-tools). 
Code for dataset pre-processing/feature extraction for Clotho dataset can
also be found [here](https://github.com/audio-captioning/clotho-dataloader).
Finally, code for handling the Clotho data (i.e. extracted features and 
one-hot encoded words) for PyTorch library (i.e. PyTorch DataLoader for
Clotho data) can also be found
[here](https://github.com/audio-captioning/clotho-dataloader).  

This repository is maintained by [K. Drossos](https://github.com/dr-costas). 


## Table of contents

  1. [Setting up the code](#setting-up-the-code)
  2. [Preparing the data](#preparing-the-data)
  3. [Use the baseline system](#use-the-baseline-system)
  5. [Explanation of settings](#explanation-of-settings)
  
## Setting up the code

To start using the audio captioning DCASE 2020 baseline system, firstly you
have to set-up the code. Please **note bold** that the code in this repository
is tested with Python 3.7.  

To set-up the code, you have to do the following: 

  1. Clone this repository.
  2. Use either pip or conda to install dependencies
  
Use the following command to clone this repository at your terminal:

````shell script
$ git clone git@github.com:audio-captioning/dacse-2020-baseline.git
````

The above command will create the directory `dacse-2020-baseline` and populate
it with the contents of this repository. The `dacse-2020-baseline` directory 
will be called root directory for the rest of this README file. 
  
For installing the dependencies, there are two ways. You can either use conda or
pip. 

### Using conda for installing dependencies

To use conda, you can issue the following command at your terminal (when you are 
in the root directory):

````shell script
$ conda create --name audio-captioning-baseline --file requirements_conda.yaml
````  

The above command will create a new environment called `audio-captioning-baseline`, which 
will have set-up all the dependencies for the audio captioning DCASE 2020 baseline.  To 
activate the `audio-captioning-baseline` environment, you can issue the following command"

````shell script
$ conda activate audio-captioning-baseline
```` 

Now, you are ready to proceed to the following steps. 

### Using pip for installing dependencies

If you do not use anaconda/conda, you can use the default Python package manager to install
the dependencies of the audio captioning DCASE 2020 baseline. To do so, you have to issue
the following command at the terminal (when you are inside the root directory): 

````shell script
$ pip install -r requirements_pip.txt
````

The above command will install the required packages using pip. Now you are ready to go
to the following steps. 

## Preparing the data

After setting-up the code for the audio captioning DCASE 2020 baseline system, you have to
obtain the Clotho dataset, place it to the proper directory, and do the feature extraction.  

### Getting the data from Zenodo

Clotho dataset is freely available online at the Zenodo platform. 
You can find Clotho at
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3490684.svg)](https://doi.org/10.5281/zenodo.3490684)
 
You should download all `.7z` files and the `.csv` files with the captions. That is, you have
do download the following files from Zenodo: 

  1. `clotho_audio_development.7z`
  2. `clotho_audio_evaluation.7z`
  3. `clotho_captions_development.csv`  
  4. `clotho_captions_evaluation.csv`
  
After downloading the files, you should place them in the `data` directory, in your root directory.

### Feature extraction

Before starting the feature extraction, you have first to expand the `7z` files. There are many 
options on how to do this. We do not want to promote different software and/or packages, so you 
can just search on Google about how to expand `7zip` files at your operating system. 

After you expand the `7z` files, you should have two directories created. The first is 
`development` and it will be created by teh `clotho_audio_development.7z` file. The second
is evaluation, and it will be created by the `clotho_audio_evaluation.7z` file.  

### Data set-up for experiments

## Use the baseline system

### Conduct an experiment

### Modify the hyper-parameters

## Explanation of settings 
