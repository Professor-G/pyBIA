.. _Machine_Learning:

Machine Learning
===========

While the convolutional neural network is the primary engine pyBIA applies for source detection, we explored the utility of other machine learning algorithms as well, of which the random forest was applied as a preliminary filter. Unlike the image classifier, the random forest model we've created takes as input numerous morphological parameters calculated from image moments. 

Given the extended emission features of Lyman-alpha Nebulae, these parameters can be used to differentiate between extended and compact objects which display no diffuse characteristics. Applying the random forest as a preliminary filter ultimately reduces the false-positive rate and optimizes the data requirements of the pipeline. 







