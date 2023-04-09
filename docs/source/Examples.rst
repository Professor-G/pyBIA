.. _examples:

Example: Gravitational Lenses
========
While we configured pyBIA for astrophysical image filtering of diffuse Lyman-alpha emission, the program modules can be called directly to create any type of image classifier using Convolutional Neural Networks (CNN). 


Model Creation
-----------

The multi-band data for 20 lens candidates can be :download:`downloaded here <lenses.npy>`.
An accompanying set of 500 images to be used for the negative class can be :download:`downloaded here <other.npy>`.

.. code-block:: python

	import numpy as np

	lenses = np.load('lenses.npy')
	other = np.load('other.npy')

The loaded arrays are 4-dimensional as per CNN convention: (num_images, img_width, img_height, img_num_channels). Note that the image size is 100x100 pixels, which is typically too large in the context of astrophysical filtering. The images are intentionally saved to be larger than ideal, so that if the data is oversampled via image augmentation techniques, the distorted, outer boundaries of the augmented image can be cropped out. To generate the classifier, initialize the ``Classifier`` class from the `cnn_model <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/cnn_model/index.html>`_  module -- note the argument ``img_num_channels``, which should be set to be the number of channels in the data (3 filters for this example -- gri).

The data is background subtracted but not normalized, which is especially important for deep-learning as the range of pixel values will directly impact the model's ability to learn, we must set our normalization parameters accordingly, which will be used to apply min-max normalization:

.. code-block:: python

	from pyBIA import cnn_model 

	model = cnn_model.Classifier(lenses, other, img_num_channels=3, normalize=True, min_pixel=0, max_pixel=[100,100,100])

Note that each class is input individually, as they will be labeled 1 (positive) and 0 (negative), accordingly. Also, ``max_pixel`` argument expects a list, containing one value per each band, as ordered in the 4-dimensional input array. In this example, the maximum pixel to min-max normalize by is set to 100 for all three channels -- therefore any pixels less than 0 will be set to 0, and any greater than 100 will be set to 100 -- after which the normalization will set the pixels to be between 0 and 1. **If your data is not normalized the gradient during backpropagation will either explode or vanish!**

Currently, pyBIA supports the implementation of three popular CNN architectures: AlexNet, ResNet-18, and VGG16. These are controlled by the ``clf`` attribute, which defaults to 'alexnet'. To create a ResNet-18 classifier, set the ``clf`` accordingly and call the ``create`` method:

.. code-block:: python

	model.clf = 'resnet18'
	model = cnn_model.create()

By default, ``verbose`` is set to 0, but this can be set to 1 to visualize the model's performance as it trains, epoch-per-epoch. The ``epochs`` attribute controls the total training epochs, which is set to 25 by default. To input validation data, set the ``val_positive`` and/or the ``val_negative`` attributes. To configure early-stopping criteria, the ``patience`` attribute can be set to a non-zero integer. This parameter will determine the number of epochs to train to after no training improvement, which could be indicative of overfitting behavior.

This CNN pipeline supports three distinct tracking metrics, configured via the ``metric`` attribute: ``binary_accuracy``, ``loss``, and ``f1_score``, as well as the validation equivalents (e.g. ``val_binary_accuracy``). In the case of ``loss``, the training will stop if this value does not improve within the designated ``patience`` number of epochs. 

Note that the ``Classifier`` does not support weighted loss functions, which are especially useful when the classes are imbalanced, as in this example. Without applying data augmentation, the only quick workaround is to enable the ``smote_sampling`` attribute, which can balance the your dataset by pixel-by-pixel interpolation of closes-neighbors (for an overview, refer to the ``Classifier`` docstring). 

.. code-block:: python

	model.smote_sampling = 1
	model = cnn_model.create(overwrite_training=True)

This will increase our 20 lense samples to 500, to match the size of the other class. Note that the ``overwrite_training`` parameter has been set to ``True``, which will replace the ``positive_class`` and ``negative_class`` class attributes with the data as it was right before the model training. This allows you to visualize the the final training data, applicable when applying data augmentation techniques. 

With ``overwrite_training`` enabled, once training is complete the ``positive_class`` attribute be assigned to be the oversampled images as synthesized by the SMOTE algorithm, which can be visualized as such:

.. code-block:: python

	model = model._plot_positive(index=0, channel=0)

This will plot the first object in the ``positive_class`` array, with the filter to displayed designated by the ``channel`` argument. If set to 'all', the figure will combine all filters to form a colorized image. 

If you wish to keep your classes imbalanced, a weighted loss function can be applied by calling the CNN model functions directly,

.. code-block:: python

	model, history = cnn_model.Alexnet(lenses, other, img_num_channels=3, loss='weighted_binary_crossentropy', weight=)

where the ``weight`` argument is a scalar factor that will control the relative weight of the positive class. When ``weight`` is greater than 1, for example, the ``loss`` function will assign more importance to the positive class, and vice versa (although in practice the positive class is the imbalanced one in binary classification problems, so it should not be less than 1). Note that setting ``weight`` equal to 1 is equivalent to using the standard binary cross-entropy loss function. 

Calling the models directly allows for maximum flexibility, as every argument is available for tuning including learning parameters, optional model callbacks and model-specific architecture arguments. For a full overview of the configurable model parameters, refer to the model-specific API documentation. 


Data Augmentation
-----------

In this example, we suffer from major class-imbalance as we have only 20 positive lenses and 500 negative others (1:50 imbalance). Applying SMOTE oversampling is unusual for images, as the images must be flattened via a single axis first, which makes this procedure extremely sensitive to object position and orientation within the image. It is thus recommended to leave the ``smote_sampling`` attribute to zero, and instead apply image augmentation techniques to classes.

The ``Classifier`` class allows you to augment your positive and/or negative data by using the following methods:

.. code-block:: python

	model.augment_positive()
	model.augment_negative()

Running these methods automatically updates the ``positive_class`` and ``negative_class`` accordingly, but as no arguments were provided, the classes will remain unchanged, with the number of augmentations to perform PER INDIVIDUAL SAMPLE determined by the ``batch`` argument (1 by default). The current API supports the following variety of augmentation routines, which must be set directly when calling the ``augment_positive`` or ``augment_negative`` methods, all disabled by default:

	**width_shift** (int): The max pixel shift allowed in either horizontal direction.
	    If set to zero no horizontal shifts will be performed. Defaults to 0 pixels.
	**height_shift** (int): The max pixel shift allowed in either vertical direction.
	    If set to zero no vertical shifts will be performed. Defaults to 0 pixels.
	**horizontal** (bool): If False no horizontal flips are allowed. Defaults to False.
	**vertical** (bool): If False no random vertical reflections are allowed. Defaults to False.
	**rotation** (int): If False no random 0-360 rotation is allowed. Defaults to False.
	**fill** (str): This is the treatment for data outside the boundaries after roration
	    and shifts. Default is set to 'nearest' which repeats the closest pixel values.
	    Can be set to: {"constant", "nearest", "reflect", "wrap"}.
	**image_size** (int, bool): The length/width of the cropped image. This can be used to remove
	    anomalies caused by the fill (defaults to 50). This can also be set to None in which case 
	    the image in its original size is returned.
	**mask_size** (int): The size of the cutout mask. Defaults to None to disable random cutouts.
	**num_masks** (int): Number of masks to apply to each image. Defaults to None, must be an integer
	    if mask_size is used as this designates how many masks of that size to randomly place in the image.
	**blend_multiplier** (float): Sets the amount of synthetic images to make via image blending.
	    Must be a ratio greater than or equal to 1. If set to 1, the data will be replaced with
	    randomly blended images, if set to 1.5, it will increase the training set by 50% with blended images,
	    and so forth. Deafults to 0 which disables this feature.
	**blending_func** (str): The blending function to use. Options are 'mean', 'max', 'min', and 'random'. 
	    Only used when blend_multiplier >= 1. Defaults to 'mean'.
	**num_images_to_blend** (int): The number of images to randomly select for blending. Only used when 
	    blend_multiplier >= 1. Defaults to 2.
	**zoom_range** (tuple): Tuple of floats (min_zoom, max_zoom) specifying the range of zoom in/out values.
	    If set to (0.9, 1.1), for example, the zoom will be randomly chosen between 90% to 110% the original 
	    image size, note that the image size thus increases if the randomly selected zoom is greater than 1,
	    therefore it is recommended to also input an appropriate image_size. Defaults to None, which disables this procedure.
	**skew_angle** (float): The maximum absolute value of the skew angle, in degrees. This is the maximum because 
	    the actual angle to skew by will be chosen from a uniform distribution between the negative and positive 
	    skew_angle values. Defaults to 0, which disables this feature.

Rotating (``rotation``), skewing (``skew_angle``), and flipping images (``horizontal`` & ``vertical``) can make the training model more robust to variations in the orientation and perspective of the input images. Likewise, shifting left/right (``widtht_shift``) and up/down (``height_shift``) will help make the model translation invariant and thus robust to the position of the object of interest within the image.

Image blending (``blend_multiplier``) can help to generate new samples through the combination of different images using a variety of blending criteria (``blend_func``). Note that by default two random images (``num_images_to_blend``) will be blended together to create one synthetic sample, and since this procedure is applied post-batch creation, the same unique sample may be randomly blended, which could be a problem if the configured augmentation parameters do not generate sufficient training feature variety.

Random cutouts (``mask_size``) can help increase the diversity of the training set and reduce overfitting, as applying this technique prevents the training model from relying too heavily on specific features of the image, thus encouraging the model to learn more general image attributes.

These techniques, when enabled, are applied in the following order:
    - Random shift + flip + rotation: Generates ``batch`` number of images.
    - Random zoom in or out.
    - If ``image_size`` is set, the image is resized so as to crop the distorted boundary.
    - Random image skewness is applied, with the ``skew_angle`` controlling the maximum angle,
        in degrees, to distort the image from its original position.
    - The batch size is now increased by a factor of ``blend_multiplier``, where each unique sample is generated
        by randomly merging ``num_images_to_blend`` together according to the blending function ``blend_func``. 
        As per the random nature, an original sample may be blended together at this stage,
        but with enough variation this may not be a problem.
    - Circular cutouts of size ``mask_size`` are randomly placed in the image, whereby
        the cutouts replace the pixel values with zeroes. Note that as per the random nature
        of the routine, if ``num_masks`` is greater than 1, overlap between each cutout may occur,
        depending on the corresponding image size to ``mask_size`` ratio.

Note that this function is used for offline data augmentation! In practice, online augmentation may be preferred as that exposes the training model to significantly more samples. If multiple channels are being used, this method will save the seeds from the augmentation of the first filter, after which the seeds will be applied to the remaining filters, thus ensuring the same augmentation procedure is applied across all channels.

For this example, we will augment each unique sample in the ``positive_class`` 25 times by setting the ``batch`` parameter, with each augmented sample generated by randomizing the enabled procedures:

.. code-block:: python
	
	batch = 25; image_size = 67
	width_shift = height_shift = 10
	vertical = horizontal = rotation = True 
	zoom_range = (0.9, 1.1)
	mask_size = num_masks = 5
	
	model.augment_positive(batch=10, width_shift=width_shift, height_shift=height_shift, vertical=vertical, horizontal=horizontal, rotation=rotation, zoom_range=zoom_range, image_size=image_size, mask_size=mask_size, num_masks=num_masks)

The ``positive_class`` will now contain 500 images so as to match our ``negative_class``. Alternatively, we could have set ``batch`` to 10, and enabled the ``blend_multiplkier`` option with a value of 2.5, to bring the final sample to 500 (20 original images times 10 augmentations times a 2.5 blending multiplier). When applying mask cutouts, it is avised to apply similar cutouts to the ``negative_class`` so as to prevent the model from associating random cutouts with the positive class:

.. code-block:: python

	model.augment_negative(mask_size=mask_size, num_masks=num_masks, image_size=image_size)

Note that the ``image_size`` paramter was set to 67 when augmenting the ``positive_class``, so even if you wish to leave the other training class the same, you would still have to resize your data by running the ``augment_negative`` method with only the ``image_size`` argument. 

As exemplified in the previous section, the ``_plot_positive`` and ``_plot_negative`` can be used for quick visualization.

.. code-block:: python

	model._plot_positive()
	model_plot_negative()

 If an image appears dark, run the methods again but manually set the ``vmin`` and ``vmax`` arguments, as by the default these limits are derived using a robust scaling. 

 To re-do the augmentations, simply reset the positive and negative class attributes and try again:

.. code-block:: python

	model.positive_class = lenses 
	model.augment_positive(blend_mulitplier=50, num_images_to_blend=3, blend_func='mean', image_size=image_size)

	model.negative_class = other 
	model.augment_negative(blend_multiplier=1, num_images_to_blend=3, blend_func='mean', image_size=image_size)

In this attempt we apply only the blending routine, note that blend_multiplier is set to 1 for the negative class, so as to implement blending for the other class while keeping the original class size the same. When the classes are ready for training, simply call the ``create`` method. 

No current options are available for augmenting the validation data, but this can be accomplished manually viat the data_augmentation.augmentation function.


Optimization
-----------

If you know what augmentation proecdures are appropriate for your dataset, but don't know what specfic thresholds to apply, you can configure the ``Classifier`` class to identify the optimal augmentation routine to apply. To enable optimization, set ``optimize`` to ``True``. The pyBIA API supports two optimization options, ``opt_aug``, which when set to ``True``, will optimize the augmentation options that have been enabled. The class attributes that control the augmentation optimization include:
	
        **batch_min** (int): The minimum number of augmentations to perform per image on the positive class, only applicable 
            if opt_aug=True. Defaults to 2.
        ** batch_max**  (int): The maximum number of augmentations to perform per image on the positive class, only applicable 
            if opt_aug=True. Defaults to 25.
        batch_other (int): The number of augmentations to perform to the other class, presumed to be the majority class.
            Defaults to 1. This is done to ensure augmentation techniques are applied consistently across both classes.        
        image_size_min (int): The minimum image size to assess, only applicable if opt_aug=True. Defaults to 50.
        image_size_max (int): The maximum image size to assess, only applicable if opt_aug=True. Defaults to 100.
        opt_max_min_pix (int, optional): The minimum max pixel value to use when tuning the normalization procedure, 
            only applicable if opt_aug=True. Defaults to None.
        opt_max_max_pix (int, optional): The maximum max pixel value to use when tuning the normalization procedure, 
            only applicable if opt_aug=True. Defaults to None.
        shift (int): The max allowed vertical/horizontal shifts to use during the data augmentation routine, only applicable
            if opt_aug=True. Defaults to 10 pixels.
        mask_size (int, optional): If enabled, this will set the pixel length of a square cutout, to be randomly placed
            somewhere in the augmented image. This cutout will replace the image values with 0, therefore serving as a 
            regularizear. Only applicable if opt_aug=True. Defaults to None.
        num_masks (int, optional): The number of masks to create, to be used alongside the mask_size parameter. If 
            this is set to a value greater than one, overlap may occur. 
        blend_max (float): A float greater than 1.1, corresponding to the increase in the minority class after the 
            blending augmentations, to be used if optimizing with opt_aug=True, then this parameter will be tuned and will be used as the 
            maximum increase to accept. For example, if opt_aug=True and blend_max=5, then the optimization will return
            an optimal value between 1 and 5. If 1, then the blending procedure is applied but the minority class size remains same the. If 5,
            then the minority class will be increased 500% via the blening routine. Defaults to 0 which disables this feature. To enable
            when opt_aug=True, set to to greater than or equal to 1.1 (a minimum of 10% increase), which would thus try different values for this
            during the optimization between 1 and 1.1.
        blend_other (float): Greater than or equal to 1. Can be zero to not apply augmentation to the majority class.
    

opt_aug=False, batch_min=2, batch_max=25, batch_other=1, balance=True, image_size_min=50, image_size_max=100, shift=10, opt_max_min_pix=None, opt_max_max_pix=None, 
        mask_size=None, num_masks=None, smote_sampling=0, blend_max=0, blending_func='mean', num_images_to_blend=2, blend_other=1, zoom_range=(0.9,1.1), skew_angle=0,
        limit_search=True, monitor1=None, monitor2=None, monitor1_thresh=None, monitor2_thresh=None, verbose=0, path=None):





