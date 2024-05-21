.. _examples:

Examples
========

While we configured pyBIA for astrophysical image filtering of diffuse Lyman-alpha emission, the program modules can be called directly to create any type of image classifier using Convolutional Neural Networks (CNN). 

Example 1: COSMOS
========

The multi-band data for 20 lens candidates can be :download:`downloaded here <lenses.npy>`.
An accompanying set of 500 images to be used for the negative class can be :download:`downloaded here <other.npy>`.

Model Creation
-----------

Start by loading the data arrays.

.. code-block:: python

   import numpy as np

   lenses = np.load('lenses.npy')
   other = np.load('other.npy')

The loaded arrays are 4-dimensional as per CNN convention: (num_images, img_width, img_height, img_num_channels). Note that the image size is 100x100 pixels, which is typically too large in the context of astrophysical filtering. The images are intentionally saved to be larger than ideal, so that if the data is oversampled via image augmentation techniques, the distorted, outer boundaries of the augmented image can be cropped out. To generate the classifier, initialize the ``Classifier`` class from the `cnn_model <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/cnn_model/index.html>`_  module -- note the argument ``img_num_channels``, which should be set to be the number of channels in the data (3 filters for this example -- gri).

The data is background subtracted but not normalized, which is especially important for deep-learning as the range of pixel values will directly impact the model's ability to learn, we must set our normalization parameters accordingly, which will be used to apply min-max normalization:

.. code-block:: python

   from pyBIA import cnn_model 

   model = cnn_model.Classifier(lenses, other, img_num_channels=3, normalize=True, min_pixel=0, max_pixel=[100,100,100])

Note that each class is input individually, as they will be labeled 1 (positive) and 0 (negative), accordingly. In this example the ``max_pixel`` argument must be a list, containing one value per each band, as ordered in final axis of the 4-dimensional input array. In this example, the maximum pixel to min-max normalize by is set to 100 for all three channels -- therefore any pixels less than 0 will be set to 0, and any greater than 100 will be set to 100 -- after which the normalization will set the pixels to be between 0 and 1. **If your data is not normalized the gradient during backpropagation will likely explode or vanish!**

Currently, pyBIA supports the implementation of two popular CNN architectures: AlexNet, and VGG16. These are controlled by the ``clf`` attribute, which defaults to 'alexnet'. To create a classifier using the AlexNet architecture, set the ``clf`` accordingly and call the ``create`` method:

.. code-block:: python

   model.clf = 'alexnet'
   model.create()

By default, the ``verbose`` attribute is set to 0 following the Keras convention, but this can be set to 1 to visualize the model's performance as it trains, epoch-per-epoch. The ``epochs`` attribute controls the total training epochs, which is set to 25 by default. To input validation data, set the ``val_positive`` and/or the ``val_negative`` attributes. To configure early-stopping criteria, the ``patience`` attribute can be set to a non-zero integer. This parameter will determine the number of epochs to stop the training at if there there is no training improvement, which would be indicative of over/under fitting behavior.

Our CNN pipeline supports three distinct tracking metrics, configured via the ``metric`` attribute: ``binary_accuracy``, ``loss``, and ``f1_score``, as well as the validation equivalents (e``val_binary_accuracy``, ``val_loss``, and ``val_f1_score``). 

Note that the ``Classifier`` does not support weighted loss functions, which are especially useful when the classes are imbalanced, as in this particular example. While data augmentation techniques are recommended in this scenario, if you wish to keep the training classes imbalanced, a weighted loss function can be applied by calling the CNN model functions directly,

.. code-block:: python

   model, history = cnn_model.AlexNet(lenses, other, img_num_channels=3, loss='weighted_binary_crossentropy', normalize=False, weight=2.0)

where the ``weight`` argument is a scalar factor that will control the relative weight of the positive class. When ``weight`` is greater than 1, for example, the ``loss`` function will assign more importance to the positive class, and vice versa (although in practice the positive class is the imbalanced one in binary classification problems, so it should not be less than 1). Note that setting ``weight`` equal to 1 is equivalent to using the standard binary cross-entropy loss function. Calling the models directly allows for maximum flexibility, as every argument is available for tuning including learning parameters, optional model callbacks and model-specific architecture arguments. For a full overview of the configurable model parameters, refer to the model-specific API documentation. 

Data Augmentation
-----------

In this example, we suffer from major class-imbalance as we have only 20 positive lenses and 500 negative others (1:25 imbalance). The example below demonstrates how to apply image augmentation techniques to create new synthetic images.

The ``Classifier`` class allows you to augment your positive and/or negative data by using the following methods:

.. code-block:: python

   model.augment_positive()
   model.augment_negative()

Running these methods automatically updates the ``positive_class`` and ``negative_class`` accordingly, but as no arguments were provided, the classes will remain unchanged. The number of augmentations to perform per individual sample is determined by the ``batch`` argument (1 by default). The current API supports the following variety of augmentation routines, which must be set directly when calling the ``augment_positive`` or ``augment_negative`` methods, all disabled by default:

-  ``width_shift`` (int): The max pixel shift allowed in either horizontal direction. If set to zero no horizontal shifts will be performed. Defaults to 0 pixels.

-  ``height_shift`` (int): The max pixel shift allowed in either vertical direction. If set to zero no vertical shifts will be performed. Defaults to 0 pixels.

-  ``horizontal`` (bool): If False no horizontal flips are allowed. Defaults to False.

-  ``vertical`` (bool): If False no random vertical reflections are allowed. Defaults to False.

-  ``rotation`` (int): If False no random 0-360 rotation is allowed. Defaults to False.

-  ``fill`` (str): This is the treatment for data outside the boundaries after roration and shifts. Default is set to 'nearest' which repeats the closest pixel values. Can be set to: {"constant", "nearest", "reflect", "wrap"}.

-  ``image_size`` (int, bool): The length/width of the cropped image. This can be used to remove anomalies caused by the fill (defaults to 50). This can also be set to None in which case the image in its original size is returned.

-  ``mask_size`` (int): The size of the cutout mask. Defaults to None to disable random cutouts.

-  ``num_masks`` (int): Number of masks to apply to each image. Defaults to None, must be an integer if mask_size is used as this designates how many masks of that size to randomly place in the image.

-  ``blend_multiplier`` (float): Sets the amount of synthetic images to make via image blending. Must be a ratio greater than or equal to 1. If set to 1, the data will be replaced with randomly blended images, if set to 1.5, it will increase the training set by 50% with blended images, and so forth. Deafults to 0 which disables this feature.
   
-  ``blending_func`` (str): The blending function to use. Options are 'mean', 'max', 'min', and 'random'. Only used when blend_multiplier >= 1. Defaults to 'mean'.
   
-  ``num_images_to_blend`` (int): The number of images to randomly select for blending. Only used when blend_multiplier >= 1. Defaults to 2.
   
-  ``zoom_range`` (tuple): Tuple of floats (min_zoom, max_zoom) specifying the range of zoom in/out values. If set to (0.9, 1.1), for example, the zoom will be randomly chosen between 90% to 110% the original image size, note that the image size thus increases if the randomly selected zoom is greater than 1, therefore it is recommended to also input an appropriate image_size. Defaults to None, which disables this procedure.
   
-  ``skew_angle`` (float): The maximum absolute value of the skew angle, in degrees. This is the maximum because the actual angle to skew by will be chosen from a uniform distribution between the negative and positive skew_angle values. Defaults to 0, which disables this feature. Using this feature is not recommended!

Rotating, skewing, and flipping images can make the training model more robust to variations in the orientation and perspective of the input images. Likewise, shifting left/right and up/down will help make the model translation invariant and thus robust to the position of the object of interest within the image. These are the recommended methods to try at first, as other techniques such as blending and applying random mask cutouts may alter the classes too dramatically.

Image blending can help to generate new samples through the combination of different images using a variety of blending criteria. Note that by default two random images will be blended together to create one synthetic sample, and since this procedure is applied post-batch creation, the same unique sample may be randomly blended, which could be problematic if the configured augmentation parameters do not generate a sufficiently varied training class. Random cutouts can help increase the diversity of the training set and reduce overfitting, as applying this technique prevents the training model from relying too heavily on specific features of the image, thus encouraging the model to learn more general image attributes. **As noted above, applying these techniques may result in an unstable classification engine as you may end up generating a synthetic class with image features that are too different, use with caution!**

These techniques, when enabled, are applied in the following order:

**(1)** Random shift + flip + rotation: Generates ``batch`` number of images.

**(2)** Random zoom in or out.

**(3)** If ``image_size`` is set, the image is resized so as to crop the distorted boundary.
    
**(4)** Random image skewness is applied, with the ``skew_angle`` controlling the maximum angle, in degrees, to distort the image from its original position.

**(5)** The batch size is now increased by a factor of ``blend_multiplier``, where each unique sample is generated by randomly merging ``num_images_to_blend`` together according to the blending function ``blend_function``. As per the random nature, an original sample may be blended together at this stage, but with enough variation this may not be a problem.

**(6)** Circular cutouts of size ``mask_size`` are randomly placed in the image, whereby the cutouts replace the pixel values with zeroes. Note that as per the random nature of the routine, if ``num_masks`` is greater than 1, overlap between each cutout may occur, depending on the corresponding image size to ``mask_size`` ratio.

Note that pyBIA's data augmentation routine is for offline data augmentation. Online augmentation may be preferred in certain cases as that exposes the training model to significantly more varied samples. If multiple image filters are being used, the data augmentation procedure will save the seeds from the augmentation of the first filter, after which the seeds will be applied to the remaining filters, thus ensuring the same augmentation procedure is applied across all channels.

For this example, we will augment each unique sample in the ``positive_class`` 25 times by setting the ``batch`` parameter, with each augmented sample generated by randomizing the enabled procedures:

.. code-block:: python
   
   batch = 25; image_size = 67
   width_shift = height_shift = 10
   vertical = horizontal = rotation = True 
   zoom_range = (0.9, 1.1)
   mask_size = num_masks = 5
   
   model.augment_positive(batch=10, width_shift=width_shift, height_shift=height_shift, vertical=vertical, horizontal=horizontal, rotation=rotation, zoom_range=zoom_range, image_size=image_size, mask_size=mask_size, num_masks=num_masks)

The ``positive_class`` will now contain 500 images so as to match our ``negative_class``. Alternatively, we could have set ``batch`` to 10, and enabled the ``blend_multiplier`` option with a value of 2.5, to bring the final sample to 500 (20 original images times 10 augmentations times a 2.5 blending multiplier). When applying mask cutouts, it is avised to apply similar cutouts to the ``negative_class`` so as to prevent the model from associating random cutouts with the positive class:

.. code-block:: python

   model.augment_negative(mask_size=mask_size, num_masks=num_masks, image_size=image_size)

Note that the ``image_size`` paramter was set to 67 when augmenting the ``positive_class``, so even if you wish to leave the other training class the same, you would still have to resize your data by running the ``augment_negative`` method with only the ``image_size`` argument. The ``_plot_positive`` and ``_plot_negative`` class attributes can be used for quick visualization. 


.. code-block:: python

   model._plot_positive()
   model._plot_negative()

If an image appears dark, run the methods again but manually set the ``vmin`` and ``vmax`` arguments, as by the default these limits are derived using a robust scaling. To re-do the augmentations, simply reset the positive and negative class attributes and try again:

.. code-block:: python

   model.positive_class = lenses 
   model.augment_positive(blend_multiplier=50, num_images_to_blend=3, blending_func='mean', image_size=image_size)

   model.negative_class = other 
   model.augment_negative(blend_multiplier=1, num_images_to_blend=3, blending_func='mean', image_size=image_size)

In this attempt we apply only the blending routine, note that blend_multiplier is set to 1 for the negative class, so as to implement blending for the other class while keeping the original class size the same. When the classes are ready for training, simply call the ``create`` method. 

No current options are available for augmenting the validation data, but this can be accomplished manually via the `data_augmentation <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/data_augmentation/index.html#pyBIA.data_augmentation.augmentation>`_ module.

Optimization
-----------

If you know what augmentation procedures are appropriate for your dataset, but don't know what specfic thresholds to use, you can configure the ``Classifier`` class to identify the optimal augmentations parameter to apply to your dataset. To enable optimization, set ``optimize`` to ``True``. This will always optimize the model learning parameters, including the learning rate, decay, optimizer, loss, activation functions, initializers and batch size). 

pyBIA supports two optimization options, one is ``opt_aug``, which when set to ``True``, will optimize the augmentation options that have been enabled. The class attributes that control the augmentation optimization include:
   
-  ``batch_min`` (int): The minimum number of augmentations to perform per image on the positive class, only applicable if opt_aug=True. Defaults to 2.

-  ``batch_max`` (int): The maximum number of augmentations to perform per image on the positive class, only applicable if opt_aug=True. Defaults to 25.

-  ``batch_other`` (int): The number of augmentations to perform to the other class, presumed to be the majority class. Defaults to 1. This is done to ensure augmentation techniques are applied consistently across both classes.        

-  ``image_size_min`` (int): The minimum image size to assess, only applicable if opt_aug=True. Defaults to 50.

-  ``image_size_max`` (int): The maximum image size to assess, only applicable if opt_aug=True. Defaults to 100.

-  ``opt_max_min_pix`` (int, optional): The minimum max pixel value to use when tuning the normalization procedure, only applicable if opt_aug=True. Defaults to None.

-  ``opt_max_max_pix`` (int, optional): The maximum max pixel value to use when tuning the normalization procedure, only applicable if opt_aug=True. Defaults to None.

-  ``shift`` (int): The max allowed vertical/horizontal shifts to use during the data augmentation routine, only applicable if opt_aug=True. Defaults to 10 pixels.

-  ``mask_size`` (int, optional): If enabled, this will set the pixel length of a square cutout, to be randomly placed somewhere in the augmented image. This cutout will replace the image values with 0, therefore serving as a regularizear. Only applicable if opt_aug=True. Defaults to None.

-  ``num_masks`` (int, optional): The number of masks to create, to be used alongside the mask_size parameter. If this is set to a value greater than one, overlap may occur. 

-  ``blend_max`` (float): A float greater than 1.1, corresponding to the increase in the minority class after the blending augmentations, to be used if optimizing with opt_aug=True, then this parameter will be tuned and will be used as the maximum increase to accept. For example, if opt_aug=True and blend_max=5, then the optimization will return an optimal value between 1 and 5. If set to 1, then the blending procedure is applied but the minority class size remains same. If set to 5, then the minority class will be increased 500% via the blening routine. Defaults to 0 which disables this feature. To enable when opt_aug=True, set to greater than or equal to 1.1 (a minimum maximum of 10% increase is required), which would thus try different values for this during the optimization between 1 and 1.1.

-  ``blend_other`` (float): Must be greater than or equal to 1. Can be set to zero to avoid applying augmentation to the majority class. It is recommended to enable this if applying blending and/or cutouts so as to avoid training a classifier that associates these features with the positive class only.
   
-  ``zoom_range`` (tuple): This sets the allowed zoom in/out range. This is not optimized, and must be set carefully according to the data being used. During the optimization, random zooms will occur according to this designated range. Can be set to zero to disable.

The second optimization routine is enabled by setting ``opt_model`` to True. This will optimize the pooling types to apply (min, max or average) as well as the main regularizer (either batch normalization or LRN) to apply to the selected CNN architcture. If ``limit_search`` is set to False, the model hyperparameters will also be optimized (individual filter information including size and stride). **It is not advised to disable the limit_search option if the augmentation procedure is also being optimized, as this will yield memory errors!**
 
The following example configures the CNN model and sets the optimization parameters. For information regarding the parameters please refer to the `documentation <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/optimization/index.html#pyBIA.optimization.objective_cnn>`_.

.. code-block:: python

   import numpy as np
   from pyBIA import cnn_model

   lenses = np.load('lenses.npy') 
   val_lenses = lenses[:4] #This will be used to validate the optimization
   lenses = lenses[4:] #Positive class training data, will be augmented
   opt_cv = 5  # Will cross-validate the positive class, since there are 20 images and 4 were selected for validation, this will create 5 models with a unique training data sample each time

   other = np.load('other.npy')

   # Model creation and optimization

   clf='alexnet' # AlexNet CNN architecture will be used 
   img_num_channels = 3 # Creating a 3-Channel model
   normalize = True # Will min-max normalize the images so all pixels are between 0 and 1

   optimize = True # Activating the optimization routine
   n_iter = 100 # Will run the optimization routine for 250 trials 
   batch_size_min, batch_size_max = 16, 64 # The training batch size will be optimized according to these bounds

   opt_model = limit_search = True # Will also optimize the CNN model architecture but with limit search on, therefore only pooling type and the regularizer are optimized
   train_epochs = 10 # Each optimization trial will train a model up to 10 epochs
   epochs = 0 # The final model will not be generated, will instead be trained post-processing
   patience = 3 # The model patience which will be applied during optimization

   opt_aug = True # Will also optimize the data augmentation procedure (positive class only by design)
   batch_min, batch_max = 10, 250 # The amount to augment EACH positive sample by, the optimal value will be selected according to this range
   shift = 5 # Will randomly shift (horizontally & vertically) each augmented image between 0 and 5 pixels
   rotation = horizontal = vertical = True # Will randomly apply rotations (0-360), and horizintal/vertical flips to each augmented image
   zoom_range = (0.9,1.1) # Will randomly apply zooming in/out between plus and minus 10% to each augmented image
   batch_other = 0 # The number of augmentations to perform to the negative class 
   balance = True # Will balance the negative class according to how many positive samples were generated during augmentation

   image_size_min, image_size_max = 50, 100 # Will try different image sizes within these bounds, the optimal value will be selected according to this range
   opt_max_min_pix, opt_max_max_pix = 10, 1500 # Will try different normalization values (the max pixel for the min-max normalization), one for each filter, the optimal value(s) will be selected according to this range

   metric = 'val_loss' # The optimzation routine will operate according to this metric's value at the end of each trial, which must also follow the patience criteria
   average = True # Will average out the above metric across all training epochs, this will be the trial value at the end

   metric2 = 'f1_score' # Optional metric that will stop trials if this doesn't improve according to the patience
   metric3 = 'binary_accuracy' # Optional metric that will stop trials if this doesn't improve according to the patience

   monitor1 = 'binary_accuracy' # Hard stop, trials will be terminated if this metric falls above the specified threshold
   monitor1_thresh = 0.99+1e-6 # Specified threshold, in this case the optimization trial will termiante if the training accuracy falls above this limit

   monitor2 = 'loss' # Hard stop, trials will be terminated if this metric falls below the specified threshold
   monitor2_thresh = 0.01-1e-6 # Specified threshold, in this case the optimization trial will termiante if the training loss falls below this limit

   model = cnn_model.Classifier(positive_class=lenses, negative_class=other, val_positive=val_lenses, img_num_channels=img_num_channels, clf=clf, normalize=normalize, optimize=optimize, n_iter=n_iter, batch_size_min=batch_size_min, batch_size_max=batch_size_max, epochs=epochs, patience=patience, metric=metric, metric2=metric2, metric3=metric3, average=average, opt_model=opt_model, train_epochs=train_epochs, opt_cv=opt_cv, opt_aug=opt_aug, batch_min=batch_min, batch_max=batch_max, batch_other=batch_other, balance=balance, image_size_min=image_size_min, image_size_max=image_size_max, shift=shift, opt_max_min_pix=opt_max_min_pix, opt_max_max_pix=opt_max_max_pix, rotation=rotation, horizontal=horizontal, vertical=vertical, zoom_range=zoom_range, limit_search=limit_search, monitor1=monitor1, monitor1_thresh=monitor1_thresh, monitor2=monitor2, monitor2_thresh=monitor2_thresh, use_gpu=True, verbose=1)

   model.create()
   model.save(dirname='Optimized_CNN_Model_CV5')

As the ``epochs`` parameter was set to zero, the final model(s) are not saved after the optimization is completed. While the models created during the optimziation routine were trained to 10 epochs as per the ``train_epochs`` parameter, this a lower limit and in principle the final model could be trained for more epochs. The ``train_epochs`` argument is set lower so as to avoid having to create full models during optimization, as this would increase the optimization time significantly. In practice, we create test models during optimization that are shallower than the final model. Nonetheless, in this example we will set the ``epochs`` to 10 as a starting point as we are dealing with a limited data set that may not require us to train the network for much more:

.. code-block:: python

   import numpy as np 
   from pyBIA import cnn_model

   lenses = np.load('lenses.npy') 
   val_lenses = lenses[:4]
   lenses = lenses[4:] 

   other = np.load('other.npy')


   model = cnn_model.Classifier(lenses, val_lenses, val_lenses)
   model.load('Optimized_CNN_Model_CV5')
   model.epochs=10 
   model.create()

