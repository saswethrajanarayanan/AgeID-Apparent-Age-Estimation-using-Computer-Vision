# AgeID-Apparent-Age-Estimation-using-Computer-Vision
UPDATE: I am working on optimising the model further and have therefore not added the training script yet.

A custom lightweight neural network that incorporates a Bag Of Visual Words model alongside a custom shallow CNN to  estimate the apparent age of a face. 

This work uses the features extracted by a Bag of Visual Words model and a custom built shallow CNN to estimate the apparent age of a face.

The Bag of Visual Words model: This model uses handcrafted features extracted by the Oriented Fast Rotated Brief (ORB) Feature Extractor, and the K-Means clustering algorithm to create histograms of image features, constituting a Bag of Visual Words. These features are learned by an ANN that concatenates with a shallow CNN. The Bag of Visual Words model
uses a K-Means model that uses a total of 100 clusters. This number for the clusters was arrived at through trial and error, training and testing the model with different cluster sizes.

The resultant model in this work is a hybrid of a shallow CNN, that extracts features through its convolutional kernels and a Bag of Visual words model backed by the ORB feature descriptor. The choice of using ORB over SIFT is attributed to ORB's speed. Both are scale invariant and using SIFT to evaluate the resultant model's performance is an exercise for another day!

The resultant models were not able to completely escape overfitting. However, on the qualitative front, good generalization was observed during the model's testing on real-time videos of faces using prediction averaging on largely sized windows of 128 frames.




