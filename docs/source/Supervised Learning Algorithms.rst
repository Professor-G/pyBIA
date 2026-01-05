.. _supervised_models:

Supervised Machine Learning
===========

.. admonition:: Under Construction (last updated 2026-01-04)

   This documentation is still being written and may change frequently!


Overview
-----------

The `ensemble_model <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/ensemble_model/index.html>`_ module includes the  `Classifier <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/ensemble_model/index.html#pyBIA.ensemble_model.Classifier>`_ class, a robust pipeline for training and optimizing popular machine learning classifiers. This module focuses on the use of more interpretable models such as tree-based estimators (Decision Trees, Random Forest, and Extra Trees Classifiers), including gradient-boosted frameworks (Gradient, Adaptive Gradient, eXtreme Gradient, and Histogram-based Gradient Boosting), simple neural networks (Multi-Layer Perceptrons), and other models such as K-Nearest Neighbors, Logistic Regression, Support Vector Classifier (including One-Class SVM), and Gaussian Naive Bayes. These models are specified via the ``clf`` argument when instantiating the class. 

This module handles data imputation (``impute`` and ``imp_method`` arguments), feature selection (``boruta_trials`` and ``boruta_model`` arguments), and hyperparameter optimization (``optimize``, ``opt_cv``, ``scoring_metric``, ``limit_search``, and ``n_iter`` arguments). The seed number (``SEED_NO``) is propagated to all downstream stochastic processes, enabling reproducibility during training and optimization. This is set to 1909 by default, and can be set to ``None`` to enable randomization at every stage.

This pipeline is flexible and can work with any training data, requiring only the feature matrix (``data_x``) and corresponding array of labels (``data_y``). In lieu of these arguments, users can also input a dataframe (``csv_file``), although this requires that a ``label`` column be present, containing the class labels. All other columns are assumed to be training features. 

Model performance, optimization results, and class separation can be visualized using built-in class methods. These include confusion matrices (``plot_conf_matrix``), ROC curves (``plot_roc_curve``), feature selection results (``plot_feature_opt``), hyperparameter optimization results (``plot_hyper_opt`` and ``plot_hyper_param_importance``), and t-SNE projections (``plot_tsne``). These are demonstrated in the example below.


Example
-----------

This example uses the per-band training data generated in the `catalog creation example <https://pybia.readthedocs.io/en/latest/source/Catalog%20Generation.html#example>`_. In the code below, these five files are combined to form a single dataframe.

.. code-block:: python

   import numpy as np
   import pandas as pd

   # Load the individual dataframes
   df_g = pd.read_csv('segm_catalog_g_band.csv')
   df_r = pd.read_csv('segm_catalog_r_band.csv')
   df_i = pd.read_csv('segm_catalog_i_band.csv')
   df_y = pd.read_csv('segm_catalog_z_band.csv')
   df_z = pd.read_csv('segm_catalog_y_band.csv')

   # The identifier columns
   exclude = ['obj_name', 'flag', 'xpix', 'ypix'] 

   # Rename the feature columns, adding the corresponding band suffix to each feature name
   new_cols_g = {col: (f"{col}_g" if col not in exclude else col) for col in df_g.columns}
   new_cols_r = {col: (f"{col}_r" if col not in exclude else col) for col in df_r.columns}
   new_cols_i = {col: (f"{col}_i" if col not in exclude else col) for col in df_i.columns}
   new_cols_z = {col: (f"{col}_z" if col not in exclude else col) for col in df_z.columns}
   new_cols_y = {col: (f"{col}_y" if col not in exclude else col) for col in df_y.columns}

   df_g = df_g.rename(columns=new_cols_g)
   df_r = df_r.rename(columns=new_cols_r)
   df_i = df_i.rename(columns=new_cols_i)
   df_z = df_z.rename(columns=new_cols_z)
   df_y = df_y.rename(columns=new_cols_y)

   # Combine and save
   df_combined = (
       df_g.merge(
           df_r, on=exclude, how='inner').merge(
           df_i, on=exclude, how='inner').merge(
           df_y, on=exclude, how='inner').merge(
           df_z, on=exclude, how='inner'
       )
   )

   df_combined.to_csv(f'merged_dataframe_five_bands.csv')


The merged catalog generated above is available for download here:

- `merged_dataframe_five_bands <https://drive.google.com/file/d/1ujBN0Qja-7TGNf96lViuxfha2l7P-Ld0/view?usp=sharing>`_


