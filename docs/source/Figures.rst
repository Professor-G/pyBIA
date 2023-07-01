.. _figures:

Figures
========


Figure 1
-----------

The multi-band data for our five confirmed lyman-alpha nebulae can be :download:`downloaded here <confirmed_diffuse.npy>`. To visuazlie the affect the sigma detection threshold has on the image segmentation object, we used the `plot_three_segm <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html#pyBIA.catalog.plot_three_segm>`_ function available in the pyBIA.catalog module.

.. code-block:: python

	import numpy as np 
	from pyBIA import catalog

	five_confirmed = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/images/final_npy/blobs_confirmed.npy')
	names = ['PRG1', 'PRG2', 'PRG3', 'PRG4', 'LABd05']

	for i in range(len(five_confirmed)):
		catalog.plot_three_segm(five_confirmed[i][:,:,0], size=100, median_bkg=0, nsig=[0.3, 0.9, 1.5], cmap='viridis', name=names[i], title='', savefig=True)


Figure 2
-----------

.. code-block:: python

	import pandas  
	import numpy as np 
	from astropy.io import fits 
	from sklearn.model_selection import cross_validate
	from pyBIA import catalog, ensemble_model

	### Create the Data Files to Generate Figure 2 ###

	data_path = '/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/Bw_FITS/'
	data_error_path = '/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/rms_images/Bw/'
	path_to_save = '/Users/daniel/Desktop/BW_TRAINING_SET_TEST/'

	#866 DIFFUSE candidates from Prescott et al. (2012) plus 3200 randomly selected OTHER objects
	training_set = pandas.read_csv('/Users/daniel/Desktop/Folders/pyBIA/pyBIA/data/training_set_objects.csv')

	sigs = np.around(np.arange(0.1, 1.51, 0.01), decimals=2)

	for sig in sigs:
		frame = [] #To store all 27 subfields
		for fieldname in np.unique(np.array(training_set['field_name'])):
			# Load the field data
			data, error_map = fits.open(data_path+fieldname+'_Bw_03_fix.fits'), fits.getdata(data_error_path+fieldname+'_Bw_03_rms.fits.fz')
			# Extract the data and corresponding ZP
			data_map, zeropoint = data[0].data, data[0].header['MAGZERO']
			# Select only the samples from this subfield
			subfield_index = np.where(training_set['field_name']==fieldname)[0]
			xpix, ypix = training_set[['xpix', 'ypix']].iloc[subfield_index].values.T
			objname, field, flag = training_set[['obj_name', 'field_name', 'flag']].iloc[subfield_index].values.T
			# Create the catalog object
			cat = catalog.Catalog(data_map, error=error_map, x=xpix, y=ypix, zp=zeropoint, nsig=sig, flag=flag, obj_name=objname, field_name=field, invert=True)
			# Generate the catalog and append the ``cat`` attribute to the frame list
			cat.create(save_file=False); frame.append(cat.cat)
		# Combine all 27 sub-catalogs into one master frame and save to the ``path_to_save``
		frame = pandas.concat(frame, axis=0, join='inner'); frame.to_csv(path_to_save+'_Bw_training_set_nsig_'+str(sig), chunksize=1000)                                                


	# Data for Figure 2

	#These are the features to use, note that the catalog includes more than this!
	columns = ['mag', 'mag_err', 'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu10', 
		'mu01', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 
		'legendre_2', 'legendre_3', 'legendre_4', 'legendre_5', 'legendre_6', 'legendre_7', 'legendre_8', 'legendre_9', 
		'area', 'covar_sigx2', 'covar_sigy2', 'covar_sigxy', 'covariance_eigval1', 'covariance_eigval2', 'cxx', 'cxy', 'cyy', 
		'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 'fwhm', 'gini', 'orientation', 'perimeter', 
		'semimajor_sigma', 'semiminor_sigma', 'max_value', 'min_value']

	rf_scores, xgb_scores = [], [] # To store the baseline accuracies as a function of sigma threshold (Left Panel of Figure 2)
	blob_nondetect, other_nondetect = [], [] # To store the number of non-detections (Right Panel of Figure 2)
	impute = True; num_cv_folds = 10 # Will impute NaN values and then assess accuracy using 10-fold CV

	for sig in sigs:
		# Load each nsig file
		#df = pandas.read_csv('/fs1/scratch/godines/BW_NSIG/training_set_nsig_'+str(sig))
		df = pandas.read_csv(path_to_save+'_Bw_training_set_nsig_'+str(sig))
		# Omit any non-detections
		mask = np.where((df['area'] != -999) & np.isfinite(df['mag']))[0]
		# Balance both classes to be of same size
		blob_index = np.where(df['flag'].iloc[mask] == 1)[0]
		other_index = np.where(df['flag'].iloc[mask] == 0)[0]
		df_filtered = df.iloc[mask[np.concatenate((blob_index, other_index[:len(blob_index)]))]]
		# Training data arrays
		data_x, data_y = np.array(df_filtered[columns]), np.array(df_filtered['flag'])
		# Create RF model first
		model = ensemble_model.Classifier(data_x, data_y, clf='rf', impute=True); model.create()
		cross_val = cross_validate(model.model, model.data_x, model.data_y, cv=num_cv_folds)
		rf_scores.append(np.mean(cross_val['test_score']))
		# Change to XGB model and re-create
		model.clf = 'xgb'; model.create()
		cross_val = cross_validate(model.model, model.data_x, model.data_y, cv=num_cv_folds)
		xgb_scores.append(np.mean(cross_val['test_score']))
		# This checks how many normalized non-detections occurred at this threshold
		blob_index, other_index = np.where(df['flag'] == 1)[0], np.where(df['flag'] == 0)[0]
		blob_nondetect.append(len(np.where(df.area.iloc[blob_index] == -999)[0]) / len(blob_index))
		other_nondetect.append(len(np.where(df.area.iloc[other_index] == -999)[0]) / len(other_index))

	score_data = np.c_[sigs, rf_scores, xgb_scores]
	non_detect_data = np.c_[sigs, blob_nondetect, other_nondetect]

	### Generate the Plots ###

	import matplotlib.pyplot as plt   
	from matplotlib.ticker import FuncFormatter

	ensemble_model._set_style_() #The custom matplotlib style

	# Figure 2 Left Panel
	max_rf_score, max_xgb_score = np.where(score_data[:,1]==np.max(score_data[:,1]))[0], np.where(score_data[:,2]==np.max(score_data[:,2]))[0] 
	optimal_index = max_xgb_score[0] if score_data[:,2][max_xgb_score] > score_data[:,1][max_rf_score] else max_rf_score[0]

	# ACCURACY PLOT
	fig, ax1 = plt.subplots()
	lns1, = ax1.plot(score_data[:,0], score_data[:,1], linestyle='--', color='b')
	lns2, = ax1.plot(score_data[:,0], score_data[:,2], linestyle='-', color='r')
	yscatter = score_data[:,2][optimal_index] if score_data[:,2][max_xgb_score] >= score_data[:,1][max_rf_score] else score_data[:,1][optimal_index]
	lns3 = ax1.scatter(score_data[:,0][optimal_index], yscatter, marker='*', s=225, edgecolors='black', c='green', alpha=0.63, label='Optimal')
	ax1.legend([lns1, lns2, lns3], ['RF', 'XGBoost', 'Optimal'], loc='upper center', ncol=3, frameon=False, handlelength=2) #r'RF $\pm$ 1$\sigma$'
	ax1.set_title('RF vs XGBoost: Baseline Performance')
	ax1.set_xlabel(r'$\sigma$ Noise Detection Limit'); ax1.set_ylabel('10-Fold CV Acc')#, size=14)
	ax1.set_xlim((0.1, 1.5)); ax1.set_ylim((0.875, 0.93))
	plt.savefig('/Users/daniel/Desktop/nsigs.png', dpi=300, bbox_inches='tight')
	#plt.show()

	# Figure 2 Right Panel

	def y_axis_formatter(x, pos):
	    return '{:.2f}'.format(round(x, 2))

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	lns1, = ax1.plot(non_detect_data[:,0], non_detect_data[:,2], linestyle='--', color='k')
	lns2, = ax2.plot(non_detect_data[:,0], non_detect_data[:,1], linestyle='-', color='k')
	lns3 = ax1.scatter(non_detect_data[:,0][optimal_index], non_detect_data[:,2][optimal_index], marker='*', s=225, edgecolors='black', c='green', alpha=0.63, label='Optimal')
	ax1.legend([lns1, lns2, lns3], ['OTHER', 'DIFFUSE', 'Optimal'], loc='upper center', ncol=3, frameon=False)
	ax1.set_title('Normalized Non-Detections')
	ax1.set_xlabel(r'$\sigma$ Noise Detection Limit')
	ax2.set_ylabel('DIFFUSE'); ax1.set_ylabel('OTHER')
	ax2.set_xlim((0.1, 1.5));ax2.set_ylim((0, 0.16)); ax1.set_ylim(0, 0.7)
	ax1.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
	ax2.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
	plt.savefig('/Users/daniel/Desktop/nsigs_nondetect.png', dpi=300, bbox_inches='tight')
	#plt.show() 



Figures 3 & 4
-----------

.. code-block:: python

	### Figures 3 and 4 ###

	import pandas
	import numpy as np
	from pyBIA import ensemble_model

	#The optimal sig threshold to apply as per Figure 2
	sig = 0.31                                                                                                                                                                                                                                
	df = pandas.read_csv('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/nsigs/BW_NSIG/BW_training_set_nsig_'+str(sig))

	# Omit any non-detections
	mask = np.where((df['area'] != -999) & np.isfinite(df['mag']))[0]

	# Balance both classes to be of same size
	blob_index = np.where(df['flag'].iloc[mask] == 1)[0]
	other_index = np.where(df['flag'].iloc[mask] == 0)[0]
	df_filtered = df.iloc[mask[np.concatenate((blob_index, other_index[:len(blob_index)]))]]

	#These are the features to use, note that the catalog includes more than this!
	columns = ['mag', 'mag_err', 'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu10', 
		'mu01', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 
		'legendre_2', 'legendre_3', 'legendre_4', 'legendre_5', 'legendre_6', 'legendre_7', 'legendre_8', 'legendre_9', 
		'area', 'covar_sigx2', 'covar_sigy2', 'covar_sigxy', 'covariance_eigval1', 'covariance_eigval2', 'cxx', 'cxy', 'cyy', 
		'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 'fwhm', 'gini', 'orientation', 'perimeter', 
		'semimajor_sigma', 'semiminor_sigma', 'max_value', 'min_value']

	# Training data arrays
	data_x, data_y = np.array(df_filtered[columns]), np.array(df_filtered['flag'])


	# Create the model object with feature and hyperparameter optimization enabled (5000 trials each)

	# Enabling 10-fold cross validation which increases the hyperparameter optimization time ten-fold
	model = ensemble_model.Classifier(data_x, data_y, clf='xgb', impute=True, optimize=True, boruta_trials=5000, n_iter=5000, opt_cv=10, limit_search=False)

	# This is how the model is created and saved afterwards
	model.create()
	model.save()

	# This is how the model can be loaded in the future
	#model = ensemble_model.Classifier(data_x, data_y, clf='xgb', impute=True, opt_cv=10)
	#model.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/models/ensemble_models/new_2500_2500')

	# Figure 3 Left Panel

	# For plotting purposes change the labels from numeric to text, although the model accepts either
	y_labels = []
	for flag in data_y:
		y_labels.append('DIFFUSE') if flag == 1 else y_labels.append('OTHER')

	# For plotting purposes, re-name the five confirmed blobs to Confirmed LyAlpha
	confirmed_names = np.loadtxt('/Users/daniel/Desktop/Folders/pyBIA/pyBIA/data/obj_name_5', dtype=str)

	for name in confirmed_names:
		index = np.where(df_filtered.obj_name == name)[0][0]
		y_labels[index] = r'Confirmed Ly$\alpha$'

	# Plotting t-SNE projection with custom y_data labels, highlighting the scatter points for the confirmed blobs
	model.plot_tsne(data_y=y_labels, special_class=r'Confirmed Ly$\alpha$', savefig=True)

	# Figure 3 Right Panel

	#Setting custom column names for plotting purposes 
	columns = [r'$B_w$ Mag', r'$B_w$ MagErr', r'$M_{00}$', r'$M_{10}$', r'$M_{01}$', r'$M_{20}$', r'$M_{11}$', r'$M_{02}$', 
		r'$M_{30}$', r'$M_{21}$', r'$M_{12}$', r'$M_{03}$', r'$\mu_{10}$', r'$\mu_{01}$', r'$\mu_{20}$', r'$\mu_{11}$', 
		r'$\mu_{02}$', r'$\mu_{30}$', r'$\mu_{21}$', r'$\mu_{12}$', r'$\mu_{03}$', r'$h_1$', r'$h_2$', r'$h_3$', r'$h_4$', 
		r'$h_5$', r'$h_6$', r'$h_7$', r'$L_2$', r'$L_3$', r'$L_4$', r'$L_5$', r'$L_6$', r'$L_7$', r'$L_8$', r'$L_9$',
		'Area', r'$\sigma^2(x)$', r'$\sigma^2(y)$', r'$\sigma^2(xy)$', r'$\lambda_1$', r'$\lambda_2$', r'$C_{xx}$', r'$C_{xy}$', r'$C_{yy}$', 
		'Eccentricity', 'Ellipticity', 'Elongation', 'Equiv. Radius', 'FWHM', 'Gini', 'Orientation', 'Perimeter', 
		r'$\sigma_{\rm major}$', r'$\sigma_{\rm minor}$', 'Max Val.', 'Min Val.']

	# Plotting only the top 25 accepted features
	model.plot_feature_opt(feat_names=columns, top=20, include_other=True, include_shadow=True, 
		include_rejected=False, flip_axes=True, save_data=False, savefig=True)

	# Figure 4 Left Panel
	 
	baseline = 0.92427745 # The maximum baseline accuracy as per Figure 2
	model.plot_hyper_opt(baseline=baseline, xlim=(1, 2500), ylim=(0.85, 0.935), xlog=True, ylog=False, savefig=True)

	# Figure 4 Right Panel 

	model.plot_hyper_param_importance(plot_time=True, savefig=True)


Figure 5
-----------

.. code-block:: python

	import pandas 
	import numpy as np
	import matplotlib.pyplot as plt 
	from pyBIA import ensemble_model 

	# Load all 2 million catalog objects and create a sub-catalog of DIFFUSE candidates #

	#The optimal sig threshold to apply as per Figure 2
	sig = 0.31                                                                                                                                                                                                                                
	df = pandas.read_csv('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/nsigs/BW_NSIG/BW_training_set_nsig_'+str(sig))

	# Omit any non-detections
	mask = np.where((df['area'] != -999) & np.isfinite(df['mag']))[0]

	# Balance both classes to be of same size
	blob_index = np.where(df['flag'].iloc[mask] == 1)[0]
	other_index = np.where(df['flag'].iloc[mask] == 0)[0]
	df_filtered = df.iloc[mask[np.concatenate((blob_index, other_index[:len(blob_index)]))]]

	#These are the features to use, note that the catalog includes more than this!
	columns = ['mag', 'mag_err', 'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03', 'mu10', 
	'mu01', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 
	'legendre_2', 'legendre_3', 'legendre_4', 'legendre_5', 'legendre_6', 'legendre_7', 'legendre_8', 'legendre_9', 
	'area', 'covar_sigx2', 'covar_sigy2', 'covar_sigxy', 'covariance_eigval1', 'covariance_eigval2', 'cxx', 'cxy', 'cyy', 
	'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 'fwhm', 'gini', 'orientation', 'perimeter', 
	'semimajor_sigma', 'semiminor_sigma', 'max_value', 'min_value']

	# Training data arrays
	data_x, data_y = np.array(df_filtered[columns]), np.array(df_filtered['flag'])

	# This is the base model, no hyperparameter optimization, uses all the features
	base_model = ensemble_model.Classifier(data_x, data_y, clf='xgb', impute=True)
	base_model.create()

	# This is the optimized model
	optimized_model = ensemble_model.Classifier(data_x, data_y, clf='xgb', impute=True)
	optimized_model.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/models/ensemble_models/new_2500_2500')

	# Load the catalog containing all 2 million other objects, extracted using sig=0.31
	other_all = pandas.read_csv('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/catalogs/other_catalog_final_031_no_dups')

	# Remove the 865 OTHER objects that are present in the training set, we will assess these individually using LoO
	other_all = other_all[~other_all['obj_name'].isin(df_filtered['obj_name'])]

	# Omit non-detections
	mask = np.where((other_all['area'] != -999) & np.isfinite(other_all['mag']))[0]
	other_all = other_all.iloc[mask]

	# Create the data_x array
	other_data_x = np.array(other_all[columns])

	# Predict all samples to create a candidates catalog
	predictions_base_model = base_model.predict(other_data_x)
	predictions_optimized_model = optimized_model.predict(other_data_x)

	# Select DIFFUSE detections (flag = 1)
	index_base = np.where(predictions_base_model[:,0] == 1)[0]
	index_optimized = np.where(predictions_optimized_model[:,0] == 1)[0]

	# Index the catalog to select only the positive detections
	candidate_catalog_base = other_all.iloc[index_base]
	candidate_catalog_optimized = other_all.iloc[index_optimized]

	# Save the probability predictions as a new columns
	candidate_catalog_base['proba'] = predictions_base_model[index_base][:,1]
	candidate_catalog_optimized['proba'] = predictions_optimized_model[index_optimized][:,1]

	# Generate the data for the histograms in Figure 5 #

	# Remove one OTHER object as the DIFFUSE will be cross-validated using LoO
	other_training = df_filtered[df_filtered.flag == 0].iloc[1:]
	diffuse_training =  df_filtered[df_filtered.flag == 1]

	# The probas of the five confirmed blobs will be saved according to their published names
	LABd05, PRG1, PRG2, PRG3, PRG4 = [],[],[],[],[]

	# To store the probas of all the other DIFFUSE objects as well as their catalog names
	all_diffuse_base_probas, all_diffuse_optimized_probas, names = [],[],[]

	#Leave-one-Out cross-validating the DIFFUSE class
	for i in range(len(diffuse_training)):
	print(i)
	# This will be the individual DIFFUSE sample to assess
	leave_one = np.array(diffuse_training[columns].iloc[i])
	# Removing this validation sample from the overall DIFFUSE training bag
	remaining = np.delete(np.array(diffuse_training[columns]), i, axis=0)
	# Setting the new training data, flag of 1 corresponds to DIFFUSE, 0 is OTHER
	data_x = np.r_[remaining, np.array(other_training[columns])]
	data_y = np.r_[[1]*len(remaining), [0]*len(other_training)]
	# Training the new base model
	new_base_model = base_model.model.fit(data_x, data_y)
	# Training the new optimized model, note that the optimized feats to use is invoked
	new_optimized_model = optimized_model.model.fit(data_x[:,optimized_model.feats_to_use], data_y)
	# Assess the left-out DIFFUSE sample using both the base and optimized models
	proba_base = new_base_model.predict_proba(leave_one.reshape(1,-1))
	proba_optimized = new_optimized_model.predict_proba(leave_one[optimized_model.feats_to_use].reshape(1,-1))
	# Save only the probability prediction that the object is DIFFUSE
	if diffuse_training.obj_name.iloc[i] == 'NDWFS_J143410.9+331730':
		LABd05.append(float(proba_base[:,1])); LABd05.append(float(proba_optimized[:,1]))
	elif diffuse_training.obj_name.iloc[i] == 'NDWFS_J143512.2+351108': 
		PRG1.append(float(proba_base[:,1])); PRG1.append(float(proba_optimized[:,1]))
	elif diffuse_training.obj_name.iloc[i] == 'NDWFS_J142623.0+351422':
		PRG2.append(float(proba_base[:,1])); PRG2.append(float(proba_optimized[:,1]))
	elif diffuse_training.obj_name.iloc[i] == 'NDWFS_J143412.7+332939':
		PRG3.append(float(proba_base[:,1])); PRG3.append(float(proba_optimized[:,1]))
	elif diffuse_training.obj_name.iloc[i] == 'NDWFS_J142653.1+343856':
		PRG4.append(float(proba_base[:,1])); PRG4.append(float(proba_optimized[:,1]))
	else:
		all_diffuse_base_probas.append(float(proba_base[:,1]))
		all_diffuse_optimized_probas.append(float(proba_optimized[:,1]))
		names.append(diffuse_training.obj_name.iloc[i])

	# The first index is the base model probability predictions, the second is the optimized model's
	five_diffuse_base_probas = np.c_[LABd05[0], PRG1[0], PRG2[0], PRG3[0], PRG4[0]][0]
	five_diffuse_optimized_probas = np.c_[LABd05[1], PRG1[1], PRG2[1], PRG3[1], PRG4[1]][0]
	five_names = ['LABd05', 'PRG1', 'PRG2', 'PRG3', 'PRG4']

	# Save the base and optimized probabilities
	np.savetxt('/Users/daniel/Desktop/LoO_Confirmed_DIFFUSE_xgb', np.c_[five_names, five_diffuse_base_probas, five_diffuse_optimized_probas], header="Names, Base_Model, Optimized_Model", fmt='%s')
	np.savetxt('/Users/daniel/Desktop/LoO_DIFFUSE_xgb', np.c_[names, all_diffuse_base_probas, all_diffuse_optimized_probas], header="Names, Base_Model, Optimized_Model", fmt='%s')

	# Repeat the same LoO process but evaluate the OTHER training for fair assessment of these objects
	# Positive detections from this LoO will be added to the candidates catalog that was created above

	# Remove one DIFFUSE object as this time the OTHER class will be cross-validated using LoO
	other_training = df_filtered[df_filtered.flag == 0]
	diffuse_training =  df_filtered[df_filtered.flag == 1].iloc[1:]

	# To store the probas of all DIFFUSE objects as well as their catalog names
	other_base_probas, other_optimized_probas, names = [],[],[]

	#Leave-one-Out cross-validating the OTHER class
	for i in range(len(other_training)):
	print(i)
	# This will be the individual OTHER sample to assess
	leave_one = np.array(other_training[columns].iloc[i])
	# Removing this validation sample from the overall OTHER training bag
	remaining = np.delete(np.array(other_training[columns]), i, axis=0)
	# Setting the new training data
	data_x = np.r_[remaining, np.array(diffuse_training[columns])]
	data_y = np.r_[[0]*len(remaining), [1]*len(diffuse_training)]
	# Training the new base model
	new_base_model = base_model.model.fit(data_x, data_y)
	# Training the new optimized model
	new_optimized_model = optimized_model.model.fit(data_x[:,optimized_model.feats_to_use], data_y)
	# Assess the left-out OTHER sample using the base and optimized model
	proba_base = new_base_model.predict_proba(leave_one.reshape(1,-1))
	proba_optimized = new_optimized_model.predict_proba(leave_one[optimized_model.feats_to_use].reshape(1,-1))
	# Save only the probability prediction that the object is DIFFUSE
	other_base_probas.append(float(proba_base[:,1]))
	other_optimized_probas.append(float(proba_optimized[:,1]))
	names.append(other_training.obj_name.iloc[i])

	# Save the base and optimized probabilities
	np.savetxt('/Users/daniel/Desktop/LoO_OTHER_xgb', np.c_[names, other_base_probas, other_optimized_probas], header="Names, Base_Model, Optimized_Model", fmt='%s')

	# Find these OTHER objects that were classified as DIFFUSE (probas greater than or equal to 50%)
	indices = []

	# Identify these positive detections
	index = np.where(np.array(other_base_probas) >= 0.5)[0]
	for name in np.array(names)[index]:
	indices.append(np.where(other_training.obj_name == name)[0][0])

	# Add to the master base candidate catalog
	df_filtered_base = other_training.iloc[indices]
	df_filtered_base['proba'] = np.array(other_base_probas)[index]
	candidate_catalog_base = pandas.concat([candidate_catalog_base, df_filtered_base], ignore_index=True)

	# Now do the same for the optimized catalog
	indices = []

	index = np.where(np.array(other_optimized_probas) >= 0.5)[0]
	for name in np.array(names)[index]:
	indices.append(np.where(other_training.obj_name == name)[0][0])

	# Add to the master optimized candidate catalog
	df_filtered_optimized = other_training.iloc[indices]
	df_filtered_optimized['proba'] = np.array(other_optimized_probas)[index]
	candidate_catalog_optimized = pandas.concat([candidate_catalog_optimized, df_filtered_optimized], ignore_index=True)

	# Save candidate catalogs
	candidate_catalog_base.to_csv('/Users/daniel/Desktop/candidate_catalog_base_xgb.csv')
	candidate_catalog_optimized.to_csv('/Users/daniel/Desktop/candidate_catalog_optimized_xgb.csv')


	# Figure 5 Left Panel -- Base Model #

	# Confusion Matrix Plot

	# Create label_y array for plotting purposes
	y_labels = []
	for flag in base_model.data_y:
	y_labels.append('DIFFUSE') if flag == 1 else y_labels.append('OTHER')

	# Assess the accuracies using 10-fold cross-validation and normalize the accuracies
	base_model.plot_conf_matrix(data_y=y_labels, k_fold=10, normalize=True, title='Base Model', savefig=True)

	# Histogram Plot
	candidate_catalog_base = pandas.read_csv('/Users/daniel/Desktop/candidate_catalog_base_xgb.csv')
	probas_candidates = np.array(candidate_catalog_base.proba)#.iloc[xxx]) #xxx = np.where(probas_candidates.area!=-999)[0]

	# Load the saved LoO data 
	confirmed_diffuse_probas = np.loadtxt('/Users/daniel/Desktop/LoO_Confirmed_DIFFUSE_xgb', dtype=str)
	all_diffuse_probas = np.loadtxt('/Users/daniel/Desktop/LoO_DIFFUSE_xgb', dtype=str)

	five_diffuse_base_probas = confirmed_diffuse_probas[:,1].astype('float')
	all_diffuse_base_probas = all_diffuse_probas[:,1].astype('float')

	# Inspecting two thresholds, 0.8 and 0.9
	index_90, index_80 = np.where(probas_candidates >= 0.9)[0], np.where(probas_candidates >= 0.8)[0]

	# Plot 
	plt.hist(probas_candidates, bins=5, weights=np.ones(len(probas_candidates)) / len(probas_candidates), color='#377eb8', label='Candidates (n='+str(len(probas_candidates))+')')
	plt.hist(all_diffuse_base_probas, bins=12, weights=np.ones(len(all_diffuse_base_probas)) / len(all_diffuse_base_probas), color='#ff7f00', alpha=0.6, label='DIFFUSE Training (n=865)')
	plt.scatter(five_diffuse_base_probas, [0.051]*len(five_diffuse_optimized_probas), marker='*', c='k', s=1000, alpha=0.72, label=r'Confirmed Ly$\alpha$ (n=5)')

	y=0.12 # Controls the position of the text
	plt.axvline(x=0.9, linestyle='--', linewidth=2, alpha=0.6, color='k', ymin=0.551)
	plt.text(0.903, 0.83+y, s=r" n(P) $\geq$ 0.9", weight="bold")
	plt.axhline(y=0.81+y, linestyle='-', linewidth=1.2, color='k', xmin=0.81, xmax=0.99)
	plt.text(0.925, 0.76+y, s=str(len(index_90)), weight="bold")

	plt.axvline(x=0.8, linestyle='--', linewidth=2, alpha=0.6, color='k', ymin=0.146)
	plt.text(0.803, 0.55+y, s=r" n(P) $\geq$ 0.8", weight="bold")
	plt.axhline(y=0.53+y, linestyle='-', linewidth=1.2, color='k', xmin=0.61, xmax=0.79)
	plt.text(0.82, 0.48+y, s=str(len(index_80)), weight="bold")

	plt.text(0.89, 0.115, s="PRG4", weight="bold")

	plt.title('XGBoost Classification Output', size=18); plt.xlabel('Probability Prediction', size=16); plt.ylabel('Normalized Counts', size=16)
	plt.xticks(ticks=[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.], 
	labels=['0.4','','0.5','','0.6','','0.7','','0.8','','0.9','','1.0'], size=14)
	plt.yticks(ticks=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0], size=14, 
	labels=['0','','0.1','','0.2','','0.3','','0.4','','0.5','','0.6','','0.7','','0.8','','0.9','','1.0'])
	plt.xlim((0.5,1.0)); plt.legend(prop={'size': 14}, loc='upper left')
	plt.savefig('/Users/daniel/Desktop/Final_Histogram_Base.png', bbox_inches='tight', dpi=300)
	plt.show()


	# Figure 5 Right Panel Histogram -- Optimized Model #

	# Confusion Matrix Plot
	optimized_model.plot_conf_matrix(data_y=y_labels, k_fold=10, normalize=True, title='Optimized Model', savefig=True)

	# Histogram Plot
	candidate_catalog_optimized = pandas.read_csv('/Users/daniel/Desktop/candidate_catalog_optimized_xgb.csv')
	probas_candidates = np.array(candidate_catalog_optimized.proba)

	five_diffuse_optimized_probas = confirmed_diffuse_probas[:,2].astype('float')
	all_diffuse_optimized_probas = all_diffuse_probas[:,2].astype('float')

	# Inspecting two thresholds, 0.8 and 0.9
	index_90, index_80 = np.where(probas_candidates >= 0.9)[0], np.where(probas_candidates >= 0.8)[0]

	# Plot
	plt.hist(probas_candidates, bins=5, weights=np.ones(len(probas_candidates)) / len(probas_candidates), color='#377eb8', label='Candidates (n='+str(len(probas_candidates))+')')
	plt.hist(all_diffuse_optimized_probas, bins=12, weights=np.ones(len(all_diffuse_optimized_probas)) / len(all_diffuse_optimized_probas), color='#ff7f00', alpha=0.6, label='DIFFUSE Training (n=865)')
	plt.scatter(five_diffuse_optimized_probas, [0.051]*len(five_diffuse_optimized_probas), marker='*', c='k', s=1000, alpha=0.72, label=r'Confirmed Ly$\alpha$ (n=5)')

	y=0.02 # Controls the position of the text
	plt.axvline(x=0.9, linestyle='--', linewidth=2, alpha=0.6, color='k', ymin=0.233)
	plt.text(0.903, 0.83+y, s=r" n(P) $\geq$ 0.9", weight="bold")
	plt.axhline(y=0.81+y, linestyle='-', linewidth=1.2, color='k', xmin=0.81, xmax=0.99)
	plt.text(0.825, 0.48+y, s=str(len(index_90)), weight="bold")

	plt.axvline(x=0.8, linestyle='--', linewidth=2, alpha=0.6, color='k', ymin=0.217)
	plt.text(0.803, 0.55+y, s=r" n(P) $\geq$ 0.8", weight="bold")
	plt.axhline(y=0.53+y, linestyle='-', linewidth=1.2, color='k', xmin=0.61, xmax=0.79)
	plt.text(0.93, 0.76+y, s=str(len(index_80)), weight="bold")

	plt.text(0.83, 0.12, s="PRG4", weight="bold")

	plt.title('XGBoost Classification Output', size=18); plt.xlabel('Probability Prediction', size=16); plt.ylabel('Normalized Counts', size=16)
	plt.xticks(ticks=[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.], 
	labels=['0.4','','0.5','','0.6','','0.7','','0.8','','0.9','','1.0'], size=14)
	plt.yticks(ticks=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0], size=14, 
	labels=['0','','0.1','','0.2','','0.3','','0.4','','0.5','','0.6','','0.7','','0.8','','0.9','','1.0'])
	plt.xlim((0.5,1.0)); plt.legend(prop={'size': 14}, loc='upper left')
	plt.savefig('/Users/daniel/Desktop/Final_Histogram_Optimized.png', bbox_inches='tight', dpi=300)
	plt.show()


Figure 6
-----------


.. code-block:: python

	### Training the CNN ### 

	# Extract Other Images #

	import os 
	import numpy as np
	import pandas as pd
	from astropy.io.fits import getdata
	from astropy.stats import SigmaClip
	from photutils.aperture import ApertureStats, CircularAnnulus
	from pyBIA.data_processing import crop_image, concat_channels 

	# Where the images will be saved (as txt files)
	bw_images_path = '/Users/daniel/Desktop/saved_images/OTHER/Bw/'
	r_images_path = '/Users/daniel/Desktop/saved_images/OTHER/R/'

	# Load the candidate catalog according to the optimized model 
	cat = pd.read_csv('/Users/daniel/Desktop/candidate_catalog_optimized_xgb.csv')

	# Select only the candidates with probability predictions greater than or equal to 85%
	index = np.where(cat.proba >= 0.85)[0]
	sample = cat.iloc[index]

	# Saving images as 120x120 pix
	image_size = 120 

	# Setting the apertures for the background subtraction, approximated using the sigma-clipped median within annuli of 20 and 35 pixel radii
	annulus_apertures = CircularAnnulus((int(image_size/2),int(image_size/2)), r_in=20, r_out=35)

	for field_name in np.unique(sample['field_name']):
		# Load the B and R broadband data
		data_bw = getdata('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/Bw_FITS/'+field_name+'_Bw_03_fix.fits')
		data_r = getdata('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/R_FITS/'+field_name+'_R_03_reg_fix.fits')
		# Select only the objects in this subfield
		subfield_index = np.where(sample['field_name'] == field_name)[0] 
		# Loop through these objects, subtract the background using aperture photometry, and save as txt file
		for i in range(len(subfield_index)):
			xpix, ypix = sample[['xpix', 'ypix']].iloc[subfield_index[i]].values.T
			# Bw first, crop the image from the entire subfield array, and save the bkg subtracted sub-array
			image = crop_image(data_bw, x=np.array(xpix), y=np.array(ypix), size=image_size, invert=True)
			bkg_stats = ApertureStats(image, annulus_apertures, error=None, sigma_clip=SigmaClip())
			np.savetxt(bw_images_path+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)
			# R next, crop the image from the entire subfield array, and save the bkg subtracted sub-array
			image = crop_image(data_r, x=np.array(xpix), y=np.array(ypix), size=image_size, invert=True)
			bkg_stats = ApertureStats(image, annulus_apertures, error=None, sigma_clip=SigmaClip())
			np.savetxt(r_images_path+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)

	# Load the object names that were saved
	obj_names = [name for name in os.listdir(bw_images_path) if 'NDWFS' in name]

	# To store the images and save as a single binary file 
	images = []

	# Load each saved file for each individual object and concat to create one single array object
	for name in obj_names:
		# Load each image individually, both filters
		Bw, R = np.loadtxt(bw_images_path+name), np.loadtxt(r_images_path+name)
		# Append as a 3D array, containing Bw-R as the third filter
		images.append(concat_channels(Bw, R, Bw-R))

	# Save the images as a 4-D array for CNN input, as well as the corresponding names
	np.save('/Users/daniel/Desktop/saved_images/xgb_output_images.npy', np.array(images))
	np.savetxt('/Users/daniel/Desktop/saved_images/xgb_output_images_names.txt', obj_names, fmt='%s')

	# Extract the DIFFUSE Images #

	confirmed_diffuse_images_path_bw = '/Users/daniel/Desktop/saved_images/confirmed_diffuse/Bw/'
	priority_diffuse_images_path_bw = '/Users/daniel/Desktop/saved_images/priority_diffuse/Bw/'
	other_diffuse_images_path_bw = '/Users/daniel/Desktop/saved_images/other_diffuse/Bw/'

	confirmed_diffuse_images_path_r = '/Users/daniel/Desktop/saved_images/confirmed_diffuse/R/'
	priority_diffuse_images_path_r = '/Users/daniel/Desktop/saved_images/priority_diffuse/R/'
	other_diffuse_images_path_r = '/Users/daniel/Desktop/saved_images/other_diffuse/R/'

	# Load the data from the Leave-one-Out cross validation analysis
	diffuse = np.loadtxt('/Users/daniel/Desktop/LoO_DIFFUSE_xgb', dtype=str)
	optimized_probas = diffuse[:,2].astype('float')

	# Select only the DIFFUSE objects that were output with probability predictions greater than 85%, this list includes the 80 priority candidates
	index = np.where(optimized_probas >= 0.85)[0]
	names_to_save = diffuse[:,0][index] 

	# The training set file
	sample = pandas.read_csv('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/nsigs/BW_NSIG/BW_training_set_nsig_0.31')

	# Will identify the priority candidates as selected by Prescott et al. (2012), so as to save separately
	obj_names_80 = np.loadtxt('/Users/daniel/Desktop/Folders/pyBIA/pyBIA/data/obj_name_80', dtype=str)

	# Will also save the five confirmed blobs
	obj_names_5 = np.loadtxt('/Users/daniel/Desktop/Folders/pyBIA/pyBIA/data/obj_name_5', dtype=str)

	for field_name in np.unique(sample['field_name']):
		# Load the B and R broadband data
		data_bw = getdata('/fs1/scratch/godines/NDWFS_Tiles/Bw/'+field_name+'_Bw_03_fix.fits')
		data_r = getdata('/fs1/scratch/godines/NDWFS_Tiles/R/'+field_name+'_R_03_reg_fix.fits')
		# Select only the objects in this subfield
		subfield_index = np.where(sample['field_name'] == field_name)[0] 
		# Loop through these objects, subtract the background using aperture photometry, and save as txt file
		for i in range(len(subfield_index)):
			if sample.obj_name.iloc[subfield_index[i]] in names_to_save or sample.obj_name.iloc[subfield_index[i]] in obj_names_5:
				xpix, ypix = sample[['xpix', 'ypix']].iloc[subfield_index[i]].values.T
				# Bw first, crop the image from the entire subfield array, and save the bkg subtracted sub-array
				image = crop_image(data_bw, x=np.array(xpix), y=np.array(ypix), size=image_size, invert=True)
				bkg_stats = ApertureStats(image, annulus_apertures, error=None, sigma_clip=SigmaClip())
				if sample.obj_name.iloc[subfield_index[i]] in obj_names_80:
					np.savetxt(priority_diffuse_images_path_bw+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)
				elif sample.obj_name.iloc[subfield_index[i]] in obj_names_5:
					np.savetxt(confirmed_diffuse_images_path_bw+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)
				else:
					np.savetxt(other_diffuse_images_path_bw+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)
				# R next, crop the image from the entire subfield array, and save the bkg subtracted sub-array
				image = crop_image(data_r, x=np.array(xpix), y=np.array(ypix), size=image_size, invert=True)
				bkg_stats = ApertureStats(image, annulus_apertures, error=None, sigma_clip=SigmaClip())
				if sample.obj_name.iloc[subfield_index[i]] in obj_names_80:
					np.savetxt(priority_diffuse_images_path_r+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)
				elif sample.obj_name.iloc[subfield_index[i]] in obj_names_5:
					np.savetxt(confirmed_diffuse_images_path_r+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)
				else:
					np.savetxt(other_diffuse_images_path_r+sample.obj_name.iloc[subfield_index[i]], image-bkg_stats.median)


	# Save the five confirmed diffuse as a single binary file #
	obj_names_confirmed_diffuse = [name for name in os.listdir(confirmed_diffuse_images_path_bw) if 'NDWFS' in name]

	images = []
	for name in obj_names_confirmed_diffuse:
		Bw, R = np.loadtxt(confirmed_diffuse_images_path_bw+name), np.loadtxt(confirmed_diffuse_images_path_r+name)
		images.append(concat_channels(Bw, R, Bw-R))

	np.save('/Users/daniel/Desktop/saved_images/confirmed_diffuse/confirmed_diffuse.npy', np.array(images))
	np.savetxt('/Users/daniel/Desktop/saved_images/confirmed_diffuse/confirmed_diffuse_names.txt', obj_names_confirmed_diffuse, fmt='%s')

	# Save the 80 priority diffuse candidates as a single binary file #
	obj_names_priority_diffuse = [name for name in os.listdir(priority_diffuse_images_path_bw) if 'NDWFS' in name]

	images = []
	for name in obj_names_priority_diffuse:
		Bw, R = np.loadtxt(priority_diffuse_images_path_bw+name), np.loadtxt(priority_diffuse_images_path_r+name)
		images.append(concat_channels(Bw, R, Bw-R))

	np.save('/Users/daniel/Desktop/saved_images/priority_diffuse/priority_diffuse.npy', np.array(images))
	np.savetxt('/Users/daniel/Desktop/saved_images/priority_diffuse/priority_diffuse_names.txt', obj_names_priority_diffuse, fmt='%s')

	# Save the other diffuse candidates as a single binary file #
	obj_names_other_diffuse = [name for name in os.listdir(other_diffuse_images_path_bw) if 'NDWFS' in name]

	images = []
	for name in obj_names_other_diffuse:
		Bw, R = np.loadtxt(other_diffuse_images_path_bw+name), np.loadtxt(other_diffuse_images_path_r+name)
		images.append(concat_channels(Bw, R, Bw-R))

	np.save('/Users/daniel/Desktop/saved_images/other_diffuse/other_diffuse.npy', np.array(images))
	np.savetxt('/Users/daniel/Desktop/saved_images/other_diffuse/other_diffuse_names.txt', obj_names_other_diffuse, fmt='%s')


	# Optimize the CNN Model #

	import numpy as np
	from pyBIA import cnn_model

	blobs = np.load('/fs1/home/godines/final_npy/blobs_confirmed.npy') 
	val_blobs = blobs[:1]
	blobs = blobs[1:]

	other = np.load('/fs1/scratch/godines/xgb_output_images.npy')
	other_test = other[:1000] # Optional test data, will be used to assess models created during the optimization routine
	other = other[1000:2000] # This will be the negative class data

	# Model creation and optimization

	clf='alexnet' # AlexNet CNN architecture will be used 
	img_num_channels = 3 # Creating a 3-Channel model
	normalize = True # Will min-max normalize the images so all pixels are between 0 and 1

	optimize = True # Activating the optimization routine
	n_iter = 250 # Will run the optimization routine for 250 trials 
	batch_size_min, batch_size_max = 16, 64 # The training batch size will be optimized according to these bounds

	opt_model = limit_search = True # Will also optimize the CNN model architecture but with limit search on, therefore only the pooling type is optimized
	train_epochs = 10 # Each optimization trial will train a model up to 10 epochs
	epochs = 0 # The final model will not be generated, will instead be trained post-processing
	patience = 3 # The model patience which will be applied during optimization
	opt_cv = 5 # Will cross-validate the positive class

	opt_aug = True # Will also optimize the data augmentation procedure (positive class only)
	batch_min, batch_max = 10, 250 # The amount to augment EACH positive sample by
	shift = 10 # Will randomly shift (horizontally & vertically) each augmented image between 0 and 10 pixels
	rotation = horizontal = vertical = True # Will randomly apply rotations (0-360), and horizintal/vertical flips to each augmented image
	zoom_range = (0.9,1.1) # Will randomly apply zooming in/out between plus and minus 10% to each augmented image
	batch_other = 0 # The number of augmentations to perform to the negative class 
	balance = True # Will balance the negative class according to how many positive samples were generated during augmentation

	image_size_min, image_size_max = 50, 100 # Will try different image sizes within these bounds 
	opt_max_min_pix, opt_max_max_pix = 10, 1500 # Will try different normalization values (the max pixel for the min-max normalization), one for each filter

	metric = 'val_loss' # The optimzation routine will operate according to this metric's value at the end of each trial, which must also follow the patience criteria
	average = True # Will average out the above metric across all training epochs, this will be the trial value at the end

	metric2 = 'f1_score' # Optional metric that will stop trials if this doesn't improve according to the patience
	metric3 = 'binary_accuracy' # Optional metric that will stop trials if this doesn't improve according to the patience

	test_acc_threshold = 0.5 # Each created model must yield accuracies greater than or equal to this value, tested against the input test_negative and/or test_positive
	post_metric = False # This test accuracy will not be used to drive the optimization 

	monitor1 = 'binary_accuracy' # Hard stop, trials will be terminated if this metric falls above the specified threshold
	monitor1_thresh = 0.99+1e-6 # Specified threshold, in this case the optimization trial will termiante if the training accuracy falls above this limit

	monitor2 = 'loss' # Hard stop, trials will be terminated if this metric falls below the specified threshold
	monitor2_thresh = 0.01-1e-6 # Specified threshold, in this case the optimization trial will termiante if the training loss falls below this limit

	model = cnn_model.Classifier(positive_class=blobs, negative_class=other, val_positive=val_blobs, img_num_channels=img_num_channels, 
		clf=clf, normalize=normalize, optimize=optimize, n_iter=n_iter, batch_size_min=batch_size_min, batch_size_max=batch_size_max, 
		epochs=epochs, patience=patience, metric=metric, metric2=metric2, metric3=metric3, average=average, test_negative=other_test, 
		test_acc_threshold=test_acc_threshold, post_metric=post_metric, opt_model=opt_model, train_epochs=train_epochs, opt_cv=opt_cv, 
		opt_aug=opt_aug, batch_min=batch_min, batch_max=batch_max, batch_other=batch_other, balance=balance, image_size_min=image_size_min, 
		image_size_max=image_size_max, shift=shift, opt_max_min_pix=opt_max_min_pix, opt_max_max_pix=opt_max_max_pix, rotation=rotation, 
		horizontal=horizontal, vertical=vertical, zoom_range=zoom_range, limit_search=limit_search, monitor1=monitor1, monitor1_thresh=monitor1_thresh, 
		monitor2=monitor2, monitor2_thresh=monitor2_thresh, use_gpu=True, verbose=1)

	model.create()
	model.save(dirname='Optimized_CNN_Model_CV5')


	# Load the optimization results and create the final model #

	import numpy as np
	from pyBIA import cnn_model

	blobs = np.load('/Users/daniel/Desktop/saved_images/confirmed_diffuse/confirmed_diffuse.npy') 
	val_blobs = blobs[:1]
	blobs = blobs[1:]

	other = np.load('/Users/daniel/Desktop/saved_images/OTHER/xgb_output_images.npy')
	other_test = other[:1000] # Optional test data, will be used to assess models created during the optimization routine
	other = other[1000:2000] # This will be the negative class data

	model = cnn_model.Classifier(blobs, other, val_blobs)
	model.load('/Users/daniel/Desktop/200gpu')
	model.epochs = 10 # Will train up to 10 epochs with the pre-loaded patience threshold
	model.create()
	model.save()


	# Plot model performance #

	import matplotlib.pyplot as plt  
	cnn_model._set_style_()

	train_metrics = np.array(model.model_train_metrics)
	val_metrics = np.array(model.model_val_metrics)
	epochs = np.arange(1, model.epochs+1)

	# Set up markers and colors for each line
	markers = ['o', 's', 'D', 'v', '^']
	colors = ['blue', 'green', 'red', 'purple', 'orange']
	names = ['PRG1', 'PRG2', 'PRG3', 'PRG4', 'LABd05']

	### Plot the f-1 score ###

	column = 2 

	# Plot the training scores
	for i in range(len(train_metrics)):
	    plt.plot(epochs, train_metrics[i][:,column], marker=markers[i], color=colors[i], label=f'Train {i+1}')

	# Plot the validation scores
	for i in range(len(val_metrics)):
	    plt.plot(epochs, val_metrics[i][:,column], marker=markers[i], linestyle='dashed', color=colors[i], label=f'Val {i+1} ({names[i]})')

	plt.xlabel('Epochs'); plt.ylabel('F1-Score')
	plt.xlim((1,10));plt.ylim((-0.01,1.01))
	plt.legend(loc='lower right', frameon=True, ncol=2)
	plt.savefig('/Users/daniel/Desktop/f1_score.png', dpi=300, bbox_inches='tight')

	### Plot the loss ###

	column = 1 

	# Plot the training scores
	for i in range(len(train_metrics)):
	    plt.plot(epochs, train_metrics[i][:,column], marker=markers[i], color=colors[i], label=f'Train {i+1}')

	# Plot the validation scores
	for i in range(len(val_metrics)):
	    plt.plot(epochs, val_metrics[i][:,column], marker=markers[i], linestyle='dashed', color=colors[i], label=f'Val {i+1} ({names[i]})')

	plt.xlabel('Epochs'); plt.ylabel('Loss')
	plt.xlim((1,10)); plt.ylim((0.007,3.5)); plt.yscale('log')
	plt.legend(loc='lower left', frameon=True, ncol=2)
	plt.savefig('/Users/daniel/Desktop/loss.png', dpi=300, bbox_inches='tight')



Figure 7
-----------

.. code-block:: python

	# Do the CNN predictions #

	# Note that the loaded objects below have already met the 85% proba prediction threshold as per the image saving procedure

	# Priority candidates as selected by Prescott et al. 2012
	priority_diffuse = np.load('/Users/daniel/Desktop/saved_images/priority_diffuse/priority_diffuse.npy')
	priority_diffuse_names = np.loadtxt('/Users/daniel/Desktop/saved_images/priority_diffuse/priority_diffuse_names.txt', dtype=str)

	# CNN prediction
	priority_diffuse_predictions = model.predict(priority_diffuse, cv_model='all', return_proba=True)

	#Save only the positive predictions from the CNN
	index = np.where(priority_diffuse_predictions[:,0] == 'DIFFUSE')[0]
	priority_diffuse = priority_diffuse[index]
	priority_diffuse_names = priority_diffuse_names[index]

	#Save in order of highests to lowest probability predictions
	priority_diffuse_probas = priority_diffuse_predictions[:,1][index]
	order = np.argsort(priority_diffuse_probas)[::-1]
	np.save('priority_diffuse_final_candidates', priority_diffuse[order])
	np.savetxt('priority_diffuse_final_candidates_names_probas', np.c_[priority_diffuse_names[order], priority_diffuse_probas[order]], fmt='%s')


	# Other diffuse candidates as selected by Prescott et al. 2012
	other_diffuse = np.load('/Users/daniel/Desktop/saved_images/other_diffuse/other_diffuse.npy') # 
	other_diffuse_names = np.loadtxt('/Users/daniel/Desktop/saved_images/other_diffuse/other_diffuse_names.txt', dtype=str)

	# CNN prediction
	other_diffuse_predictions = model.predict(other_diffuse, cv_model='all', return_proba=True)

	#Save only the positive predictions from the CNN
	index = np.where(other_diffuse_predictions[:,0] == 'DIFFUSE')[0]
	other_diffuse = other_diffuse[index]
	other_diffuse_names = other_diffuse_names[index]

	#Save in order of highests to lowest probability predictions
	other_diffuse_probas = other_diffuse_predictions[:,1][index]
	order = np.argsort(other_diffuse_probas)[::-1]
	np.save('other_diffuse_final_candidates', other_diffuse[order])
	np.savetxt('other_diffuse_final_candidates_names_probas', np.c_[other_diffuse_names[order], other_diffuse_probas[order]], fmt='%s')


	# The OTHER candidates as selected by the XGBoost classifier
	other_candidates = np.load('/Users/daniel/Desktop/saved_images/OTHER/xgb_output_images.npy')
	other_candidates_names = np.loadtxt('/Users/daniel/Desktop/saved_images/OTHER/xgb_output_images_names.txt', dtype=str)

	# CNN prediction
	other_candidates_predictions = model.predict(other_candidates, cv_model='all', return_proba=True)

	#Save only the positive predictions from the CNN
	index = np.where(other_candidates_predictions[:,0] == 'DIFFUSE')[0]
	other_candidates = other_candidates[index]
	other_candidates_names = other_candidates_names[index]

	#Save in order of highests to lowest probas
	other_candidate_probas = other_candidates_predictions[:,1][index]
	order = np.argsort(other_candidate_probas)[::-1]
	np.save('OTHER_final_candidates', other_candidates[order])
	np.savetxt('OTHER_final_candidates_names_probas', np.c_[other_candidates_names[order], other_candidate_probas[order]], fmt='%s')


	### Color-Color ###

	import pandas 
	import numpy as np

	# Load the candidate catalog (~54k objects)
	csv_candidates = pandas.read_csv('/Users/daniel/Desktop/candidate_catalog_optimized_xgb.csv') 

	# Load the names and probabilities of the candidates that were positively classified by the CNN
	candidate_names_probas = np.loadtxt('OTHER_final_candidates_names_probas', dtype=str)

	# Index the csv to only these positive candidates
	candidates_indices = []
	for i in range(len(csv_candidates)):
		if csv_candidates.obj_name.iloc[i] in candidate_names_probas[:,0]:
			candidates_indices.append(i)

	csv_candidates = csv_candidates.iloc[candidates_indices]

	# Load the diffuse training objects 
	sig = 0.31                                                                                                                                                                                                                                
	training_set = pandas.read_csv('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/nsigs/BW_NSIG/BW_training_set_nsig_'+str(sig))
	blob_index = np.where(training_set['flag'] == 1)[0] # Select only the diffuse objects
	training_set = training_set.iloc[blob_index]

	# Will load the names of the five confirmed blobs to create a subsample dataframe, will be used for color-color selection
	confirmed_diffuse_names = np.loadtxt('/Users/daniel/Desktop/Folders/pyBIA/pyBIA/data/obj_name_5', dtype=str)

	confirmed_diffuse_indices = []
	for i in range(len(training_set)):
		if training_set.obj_name.iloc[i] in confirmed_diffuse_names:
			confirmed_diffuse_indices.append(i)

	confirmed_set = training_set.iloc[confirmed_diffuse_indices]

	# Now load the names of the diffuse training objects selected by the CNN, not including the confirmed blobs
	priority_diffuse_names_probas = np.loadtxt('priority_diffuse_final_candidates_names_probas', dtype=str)
	other_diffuse_names_probas = np.loadtxt('other_diffuse_final_candidates_names_probas', dtype=str)

	diffuse_indices = []
	for i in range(len(training_set)):
		if training_set.obj_name.iloc[i] in np.r_[priority_diffuse_names_probas[:,0], other_diffuse_names_probas[:,0]]:
			diffuse_indices.append(i)

	training_set = training_set.iloc[diffuse_indices]

	# Combine the two dataframes, this is the Bw band, doesn't include the five confirmed
	final_candidate_catalog_bw = pandas.concat([csv_candidates, training_set], ignore_index=True)
	final_candidate_catalog_bw.to_csv('_Bw_final_candidate_catalog.csv', chunksize=1000)

	# Save a dataframe with only the confirmed blobs, to be used for the color-color selection below
	confirmed_set.to_csv('_Bw_final_confirmed_catalog.csv')


	# Create a new catalog in the R band for the final candidates
	from pyBIA import catalog  
	from astropy.io import fits 

	data_path = '/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/R_FITS/'
	data_error_path = '/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/rms_images/R/npy/'

	sig = 0.31
	frame = [] #To store all 27 subfields
	for fieldname in np.unique(np.array(final_candidate_catalog_bw['field_name'])):
		# Load the field data
		data, error_map = fits.open(data_path+fieldname+'_R_03_reg_fix.fits'), np.load(data_error_path+fieldname+'_R_03_rms.npy')
		# Extract the data and corresponding ZP
		data_map, zeropoint = data[0].data, data[0].header['MAGZERO']
		# Select only the samples from this subfield
		subfield_index = np.where(final_candidate_catalog_bw['field_name']==fieldname)[0]
		xpix, ypix = final_candidate_catalog_bw[['xpix', 'ypix']].iloc[subfield_index].values.T
		objname, field, flag = final_candidate_catalog_bw[['obj_name', 'field_name', 'flag']].iloc[subfield_index].values.T
		# Create the catalog object
		cat = catalog.Catalog(data_map, error=error_map, x=xpix, y=ypix, zp=zeropoint, nsig=sig, flag=flag, obj_name=objname, field_name=field, invert=True)
		# Generate the catalog and append the ``cat`` attribute to the frame list
		cat.create(save_file=False); frame.append(cat.cat)
	# Combine all 27 sub-catalogs into one master frame and save
	frame = pandas.concat(frame, axis=0, join='inner'); frame.to_csv('_R_final_candidate_catalog.csv', chunksize=1000)                                                

	# Create a new catalog in the R band for the five confirmed blobs
	from pyBIA import catalog  
	from astropy.io import fits 

	data_path = '/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/R_FITS/'
	data_error_path = '/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/data_files/NDWFS_Tiles/rms_images/R/npy/'

	sig = 0.31
	frame = [] #To store all 27 subfields
	for fieldname in np.unique(np.array(confirmed_set['field_name'])):
		# Load the field data
		data, error_map = fits.open(data_path+fieldname+'_R_03_reg_fix.fits'), np.load(data_error_path+fieldname+'_R_03_rms.npy')
		# Extract the data and corresponding ZP
		data_map, zeropoint = data[0].data, data[0].header['MAGZERO']
		# Select only the samples from this subfield
		subfield_index = np.where(confirmed_set['field_name']==fieldname)[0]
		xpix, ypix = confirmed_set[['xpix', 'ypix']].iloc[subfield_index].values.T
		objname, field, flag = confirmed_set[['obj_name', 'field_name', 'flag']].iloc[subfield_index].values.T
		# Create the catalog object
		cat = catalog.Catalog(data_map, error=error_map, x=xpix, y=ypix, zp=zeropoint, nsig=sig, flag=flag, obj_name=objname, field_name=field, invert=True)
		# Generate the catalog and append the ``cat`` attribute to the frame list
		cat.create(save_file=False); frame.append(cat.cat)
	# Combine all 27 sub-catalogs into one master frame and save
	frame = pandas.concat(frame, axis=0, join='inner'); frame.to_csv('_R_final_confirmed_catalog.csv')                                                



	# Plot #
	import pandas as pd
	import matplotlib.pyplot as plt  
	from pyBIA.cnn_model import _set_style_

	# Load the dataframes, note that the Bw and R csvs do not correspond 1-1, need to sort by obj_name
	final_candidate_catalog_bw = pd.read_csv('_Bw_final_candidate_catalog.csv')
	final_candidate_catalog_r = pd.read_csv('_R_final_candidate_catalog.csv')

	# Sort both dataframes alphabetically by the 'obj_name' column
	final_candidate_catalog_bw.sort_values('obj_name', inplace=True)
	final_candidate_catalog_r.sort_values('obj_name', inplace=True)

	# Reset the indices of both dataframes
	final_candidate_catalog_bw.reset_index(drop=True, inplace=True)
	final_candidate_catalog_r.reset_index(drop=True, inplace=True)

	final_confirmed_catalog_bw = pd.read_csv('_Bw_final_confirmed_catalog.csv')
	final_confirmed_catalog_r = pd.read_csv('_R_final_confirmed_catalog.csv')

	# Sort both dataframes by the 'obj_name' column
	final_confirmed_catalog_bw.sort_values('obj_name', inplace=True)
	final_confirmed_catalog_r.sort_values('obj_name', inplace=True)

	# Reset the indices of both dataframes
	final_confirmed_catalog_bw.reset_index(drop=True, inplace=True)
	final_confirmed_catalog_r.reset_index(drop=True, inplace=True)

	_set_style_()

	plt.scatter(final_confirmed_catalog_bw.mag - final_confirmed_catalog_r.mag, final_confirmed_catalog_bw.area, marker='*', c='red', edgecolors='black', s=300, alpha=0.95, label=r'Confirmed Ly$\alpha$')
	plt.scatter(final_candidate_catalog_bw.mag - final_candidate_catalog_r.mag, final_candidate_catalog_bw.area, marker='.', c='black', s=25, alpha=0.06, label=r'Other Candidates')
	plt.xlabel('BW - R', size=18)
	plt.ylabel('Area', size=18)
	plt.title('Color Cut Final Candidates (n=10299)', size=20)
	#plt.ylim((400,2000)); plt.xlim((-0.6, 0.8))
	#plt.xscale('log')
	#plt.yscale('log')
	plt.legend()

	plt.show()


	index_color = np.where( ((final_candidate_catalog_bw.mag - final_candidate_catalog_r.mag) <= 0.8) & ( (final_candidate_catalog_bw.mag - final_candidate_catalog_r.mag) >= -0.6))[0]
	index_area = np.where( ((final_candidate_catalog_bw.area - final_candidate_catalog_r.area)[index_color] <= 2000) & ( (final_candidate_catalog_bw.area - final_candidate_catalog_r.area)[index_color] >= 400))[0]
	index = index_color[index_area]

	plt.scatter(final_confirmed_catalog_bw.mag - final_confirmed_catalog_r.mag, final_confirmed_catalog_bw.area, marker='*', c='red', edgecolors='black', s=300, alpha=0.95, label=r'Confirmed Ly$\alpha$')
	plt.scatter(final_candidate_catalog_bw.mag.iloc[index] - final_candidate_catalog_r.mag.iloc[index], final_candidate_catalog_bw.area.iloc[index_color_and_area], marker='.', c='black', s=25, alpha=0.06, label=r'Other Candidates')
	plt.xlabel('BW - R', size=18)
	plt.ylabel('Area', size=18)
	plt.title('Color Cut Selected (n=2034)', size=20)
	#plt.xscale('log')
	#plt.yscale('log')
	plt.legend()
	plt.show()




