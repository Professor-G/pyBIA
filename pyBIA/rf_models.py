#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:04:23 2021

@author: daniel
"""
import pandas
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, auc, RocCurveDisplay
from sklearn.model_selection import KFold, StratifiedKFold

from skopt import BayesSearchCV, plots
from skopt.plots import plot_convergence, plot_objective
from pyBIA.optimization import hyper_opt, boruta_opt, KNN_imputation, MissForest_imputation

random.seed(41)

blob = pandas.read_csv('/Users/daniel/Desktop/diffuse_catalog')
other = pandas.read_csv('/Users/daniel/Desktop/other_catalog')
cols = other.columns.values[8:] #Remove index cols, REMEMBER to set index=False.

blob = blob[cols]
other = other[cols]

mask = np.where(other.area != -999)[0]
other_ = other.iloc[mask]
rand_inx = [int(i) for i in random.sample(range(0, len(mask)), len(blob))]

other_train = other.iloc[rand_inx]
blob_train = blob

labels_blob = np.array(['DIFFUSE']*len(blob_train))
labels_other = np.array(['OTHER']*len(other_train))

data_x = np.concatenate((blob_train, other_train))
data_y = np.r_[labels_blob, labels_other]

#training_set = strawman_imputation(training_set)
#training_set[training_set>1e6] = 1e6
#training_set[training_set<1e-6] = 1e-6
#rf = RandomForestClassifier(n_estimators=1000)

##############
def create(data_x, data_y, impute=True, optimize=True, imp_method='MissForest'):
    """Creates the Random Forest model and PCA transformation used for classification.
    
    Example:
        If the impute and optimize arguments are set to False, only the Random Forest
        classifier will be returned.

        >>> classifier = create(data_x, data_y, impute=False, optimize=False)

        If impute=True, the imputer will also be returned which is necessary to 
        properly transform unseen, new data.

        >>> classifier, imputer = create(data_x, data_y, impute=True, optimize=False)

        Lastly, if optimize=True, the index of the useful features is also returned.

        >>> classifier, index = create(data_x, data_y, impute=False, optimize=True)

        By default, impute and optimize are set to True, therefore three items are returned.

        >>> classfier, imputer, index = create(data_x, data_y)

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        impute (bool): If False no data imputation will be performed. Defaults to True,
            which will result in two outputs, the classifier and the imputer to save
            for future transformations. 
        optimize (bool): If True the Boruta algorithm will be run to identify the features
            that contain useful information, after which the optimal Random Forest hyperparameters
            will be calculated using Bayesian optimization. 
        imp_method (str): The imputation techinque to apply, can either be 'KNN' for k-nearest
            neighbors imputation, or 'MissForest' for the MissForest machine learning imputation
            algorithm. Defaults to 'MissForest'.
    
    Returns:
        Random Forest classifier model created with scikit-learn. If optimize=True, this
        model will already include the optimal hyperparameters. 
    """
    
    model = RandomForestClassifier()
    if impute is False and optimize is False:
        model.fit(data_x, data_y)
        return model 

    if impute:
        if imp_method == 'KNN':
            data_x, imputer = KNN_imputation(data=data_x, imputer=None)
        elif imp_method == 'MissForest':
            data_x, imputer = MissForest_imputation(data=data_x, imputer=None)
        elif imp_method == 'Strawman':
            raise ValueError('Strawman imputation method not currently supported, set imp_method to either "KNN" or "MissForest"')
        else:
            raise ValueError('Invalid imputation method, currently only "KNN" and "MissForest" arguments are supported.')
        
        if optimize is False:
            model.fit(data_x, data_y)
            return model, imputer 

    features_index = boruta_opt(model, data_x, data_y)
    model, best_params = hyper_opt(model, data_x[:,features_index], data_y)
    print('Hyperparameter optimization complete! Optimal parameters: \
        {}'.format(best_params))
    model.fit(data_x[:,features_index], data_y)

    return model, imputer, features_index

def plot_conf_matrix(data_x, data_y, k_fold=10, normalize=True, classes=["DIFFUSE","OTHER"], title='Confusion matrix'):
    """
    Returns a confusion matrix with k-fold validation.

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Defaults to a list containing 'DIFFUSE' & 'OTHER'. 
        title (str, optional): The title of the output plot. 

    Returns:
        AxesImage.
    """

    if len(classes) != 2:
        raise ValueError('Only two training classes are currently supported.')
    
    predicted_target, actual_target = evaluate_model(data_x, data_y, normalize=normalize, k_fold=k_fold)
    generate_matrix(predicted_target, actual_target, normalize=normalize, classes=classes, title=title)

def plot_roc_curve(classifier, data_x, data_y, k_fold=10, title="Receiver Operating Characteristic Curve"):
    """
    Plots ROC curve with 10k fold validation, as such the 
    standard deviation variations are plotted.
    
    Example:
        To assess the performance of a random forest classifier (created
        using the scikit-learn implementation) we can run the following:
        
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> classifier = RandomForestClassifier()
        >>> plot_roc_curve(classifier, data_x, data_y)
    
    Args:
        classifier: The machine learning classifier to optimize.
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.
        title (str, optional): The title of the output plot. 
    
    Returns:
        AxesImage
    """
    
    cv = StratifiedKFold(n_splits=k_fold)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    train = data_x
    fig, ax = plt.subplots()
    for i, (data_x, test) in enumerate(cv.split(train, data_y)):
        classifier.fit(train[data_x], data_y[data_x])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            train[test],
            data_y[test],
            name="ROC fold {}".format(i+1),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[0, 1.0],
        ylim=[0.0, 1.0],
        title="Receiver Operating Characteristic Curve",
    )
    ax.legend(loc="lower right")
    plt.ylabel('True Positive Rate', size=14)
    plt.xlabel('False Positive Rate', size=14)
    plt.title(label=title,fontsize=18)
    plt.show()

def evaluate_model(data_x, data_y, normalize=True, k_fold=10):
    """
    Cross-checks model accuracy and outputs both the predicted
    and the true class labels. 

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.

    Returns:
        The first output is the 1D array of the true class labels.
        The second output is the 1D array of the predicted class labels.
    """

    k_fold = KFold(k_fold, shuffle=True, random_state=1)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train_ix, test_ix in k_fold.split(data_x):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
        # Fit the classifier
        classifier = RandomForestClassifier()#bootstrap=False, criterion='entropy', max_depth=38, max_features='sqrt', max_leaf_nodes=167, min_samples_leaf=9, min_samples_split=4, n_estimators=1000) #866
        classifier.fit(train_x, train_y)
        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)
        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets

def generate_matrix(predicted_labels_list, actual_targets, normalize=True, classes=["DIFFUSE","OTHER"], title='Confusion matrix'):
    """
    Generates the confusion matrix using the output from the evaluate_model() function.

    Args:
        predicted_labels_list: 1D array containing the predicted class labels.
        actual_targets: 1D array containing the actual class labels.
        normalize (bool, optional): If True the matrix accuracy will be normalized
            and displayed as a percentage accuracy. Defaults to True.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Defaults to a list containing 'DIFFUSE' & 'OTHER'. 
        title (str, optional): The title of the output plot. 

    Returns:
        AxesImage.
    """

    conf_matrix = confusion_matrix(actual_targets, predicted_labels_list)
    np.set_printoptions(precision=2)

    plt.figure()
    if normalize == True:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title='Confusion matrix, without normalization')
    elif normalize == False:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title='Normalized Confusion Matrix')
    plt.show()

def generate_plot(conf_matrix, classes, normalize=False, title='Confusion Matrix'):
    """
    Generates the confusion matrix figure object, but does not plot.
    
    Args:
        conf_matrix: The confusion matrix generated using the generate_matrix() function.
        classes (list): A list containing the label of the two training bags. This
            will be used to set the axis. Defaults to a list containing 'DIFFUSE' & 'OTHER'. 
        normalize (bool, optional): If True the matrix accuracy will be normalized
            and displayed as a percentage accuracy. Defaults to True.
        title (str, optional): The title of the output plot. 

    Returns:
        AxesImage object. 
    """

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title, fontsize=20)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predicted label',fontsize=16)

    return conf_matrix
