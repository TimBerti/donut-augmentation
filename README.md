# Periodic Boundary Condition Image Augmentation for CNNs

## Overview

This project introduces an approach to image augmentation for Convolutional Neural Networks (CNNs) using periodic boundary conditions. By treating images as if they were mapped onto a torus, this method allows for seamless shifts and rotations without loss of information, enhancing the robustness of CNN models to positional variances.

## Background

Traditional image augmentation techniques such as random cropping, rotation, and flipping can lead to the loss of critical features when objects of interest are near the edges of the image. This method aims to address this limitation by applying a concept inspired by periodic boundary conditions used in computational simulations. This ensures that all parts of an image are treated equally during the augmentation process, thereby preventing information loss and improving model generalization.

## Implementation

The key component of this approach is the implementation of periodic boundary conditions within the image preprocessing pipeline. Specifically, we extend the image in all directions, effectively mapping it onto the surface of a torus. This mapping allows any shifted version of the image to retain all original content, with parts that go beyond one edge reappearing on the opposite one.
A periodic padding layer in the model ensures that convolution operations respect the topology of the torus, enabling seamless transitions across image boundaries.

### Shift with Periodic Boundary Conditions

This custom transformation applies random shifts to the image in both horizontal and vertical directions, with the shifted parts wrapping around the edges, maintaining the continuity of features.

## Results

Preliminary tests on the MNIST dataset have demonstrated the potential of this approach in enhancing the performance and generalization capabilities of CNN models, particularly in scenarios where object position variability is a significant factor.
