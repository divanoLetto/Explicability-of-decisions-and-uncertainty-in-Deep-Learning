# Explicability of decisions and uncertainty in Deep Learning
## Authors: Lorenzo Mandelli
#### Università degli Studi di Firenze | UiT The Arctic University of Norway 

---
![](https://img.shields.io/github/contributors/divanoletto/Explicability-of-decisions-and-uncertainty-in-Deep-Learning?color=light%20green) ![](https://img.shields.io/github/repo-size/divanoletto/Explicability-of-decisions-and-uncertainty-in-Deep-Learning)
---

## Introduction

One of the problems that exist in deep learning is the ability to **explain why an AI makes a particular decision**. <br/>
The models that through a series of computations produce a decision are treated as a black box: they are not able to give explanations on why they assume a certain behavior.<br/>
It is therefore useful to be able to elaborate methods that, given a model, are able to provide an interpretation of its behavior.<br/>
In this project we implemented methods that allow to explain the behavior of a model as a whole (**global methods**) or with respect to a specific decision (**local methods**).

## Methods

- **Model Class Visualization:** <br/>
For each possible prediction class (for example Dog, Cat, Snake, ...) it is possible to have the model generate the image that maximizes the probability of belonging to that class. We question the model about its image archetype of a particular class.<br/>
To improve the quality of the images produced every k iterations a Gaussian kernel is applied to the result.

- **Class Saliency Maps:** <br/>
Given an image and a prediction, it is possible to interrogate the model on which pixels it has concentrated on for its decision. <br/>
This gives us spatial information on the reasons of its decision. <br/>
To obtain a less noisy response the guided backpropagation of the ReLu functions has been implemented. 

Real Image            |  Class Saliency Map   |  CLass Saliency Map with Guided Backpropagation
:-------------------------:|:-------------------------:|:---------------------------------:
![](https://github.com/divanoLetto/Explicability-of-decisions-and-uncertainty-in-Deep-Learning/blob/master/images/2_real.JPEG)  |  ![](https://github.com/divanoLetto/Explicability-of-decisions-and-uncertainty-in-Deep-Learning/blob/master/images/2_csm.JPEG)  |  ![](https://github.com/divanoLetto/Explicability-of-decisions-and-uncertainty-in-Deep-Learning/blob/master/images/2_csm_g.JPEG)

- **Uncertainties via Monte Carlo Dropout:**<br/>
We answer the question how reliable a prediction of a model is by calculating the variance of its decision by randomly sampling its weights from the distribution of possible weights that is possible to obtain by activating the Dropout layers.

## Installation

The following libraries are needed to run the code:

1. torch (version =  )
2. torchvision (version = )
3. matplotlib (version =  )
4. scikit-learn (version =  )
5. numpy 
6. scipy 

## Data preprocessing

The dataset is made up of 91 images each of a different class. 
In order to preprocess the dataset it is necessary to run the process_data.py file which splits the images into folders according to their class.

## Usage

In order to run the programs files for the first time using the default settings (those that yield the best results), simply run the following scripts:

1. 2b 
2. 2c
3. 2d
4. 2e
5. 2f
6. 2h
