from astropy.table import Table
import numpy as np
import cv2

def make_moments_table(image):
	"""
	This function takes a 2D image array as input and 
	calculates the image moments, central moments, Hu moments, 
	and geometrically centered moments to third order. The fourier descriptors
	are also computed but only the first three are kept by default.
	These features are concatenated and then saved in an astropy Table.
	This is the main function the is used to generating the catalog (see pyBIA.catalog.morph_parameters)

	Args:
		image (ndarray): A 2D array representing an image.

	Returns:
		A astropy table with 40 columns, one for each moment or descriptor."
    """

	if len(image.shape) != 2:
		raise ValueError("Input image must be 2D.")
  
	moments, central_moments, hu_moments = calculate_moments(image), calculate_central_moments(image), calculate_hu_moments(image)
	geometrically_centered_moments, fourier_descriptors = calculate_geometrically_centered_moments(image), calculate_fourier_descriptors(image, k=3)
	
	features = moments + central_moments + hu_moments + fourier_descriptors + geometrically_centered_moments

	# The catalog mu00 and g00 because these are equivalent to m00! BUT these are kept here because these col_names must 
	# match what the functions output
	col_names = ['m00','m10','m01','m20','m11','m02','m30','m21','m12','m03',
		'mu00', 'mu10','mu01','mu20','mu11','mu02','mu30','mu21','mu12','mu03',
		'hu1','hu2','hu3','hu4','hu5','hu6','hu7', 'fourier_1','fourier_2','fourier_3',
		'g00', 'g10', 'g01', 'g20','g11','g02','g30','g21','g12','g03']

	features, col_names = np.array(features), np.array(col_names)
	dtype = ('f8',) * len(col_names)
	features_table = Table(data=features, names=col_names, dtype=dtype)
	
	return features_table

def calculate_moments(image):
	"""
	This function takes a 2D image array as input 
	and calculates the raw image moments to third order.

	Args:
		image (ndarray): A 2D array representing an image.

	Returns: 
		A tuple of 10 values representing the calculated image moments (m00, m10, m01, m20, m11, m02, m30, m21, m12, m03)
	"""

	if len(image.shape) != 2:
		raise ValueError("Input image must be 2D.")
    
	rows, cols = image.shape
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))

	m00 = np.sum(image)
	m10 = np.sum(x * image)
	m01 = np.sum(y * image)
	m20 = np.sum((x**2) * image)
	m11 = np.sum(x * y * image)
	m02 = np.sum((y**2) * image)
	m30 = np.sum((x**3) * image)
	m21 = np.sum((x**2 * y) * image)
	m12 = np.sum((x * y**2) * image)
	m03 = np.sum((y**3) * image)

	return [m00, m10, m01, m20, m11, m02, m30, m21, m12, m03]

def calculate_central_moments(image):
	"""
	This function takes a 2D image array as input and 
	calculates the central moments to third order.

	Args:
		image (ndarray): A 2D array representing an image.

	Returns:
		A tuple of 10 values representing the calculated central moments (mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03)
	"""

	if len(image.shape) != 2:
		raise ValueError("Input image must be 2D.")
    
	rows, cols = image.shape
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))

	m00 = np.sum(image)
	x_bar = np.sum(x * image) / m00
	y_bar = np.sum(y * image) / m00

	mu00 = m00
	mu10 = np.sum((x - x_bar) * image)
	mu01 = np.sum((y - y_bar) * image)
	mu20 = np.sum((x - x_bar)**2 * image)
	mu11 = np.sum((x - x_bar) * (y - y_bar) * image)
	mu02 = np.sum((y - y_bar)**2 * image)
	mu30 = np.sum((x - x_bar)**3 * image)
	mu21 = np.sum((x - x_bar)**2 * (y - y_bar) * image)
	mu12 = np.sum((x - x_bar) * (y - y_bar)**2 * image)
	mu03 = np.sum((y - y_bar)**3 * image)

	return [mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03]

def calculate_hu_moments(image):
	"""
	This function takes a 2D image array as 
	input and calculates the 7 Hu moments.

	Args:
		image (ndarray): A 2D array representing an image.

	Returns:
		A tuple of 7 values representing the calculated Hu moments (hu1, hu2, hu3, hu4, hu5, hu6, hu7)
	"""

	if len(image.shape) != 2:
		raise ValueError("Input image must be 2D.")

	mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03 = calculate_central_moments(image)
	s = np.sqrt(mu20 + mu02)

	hu1 = mu20 + mu02
	hu2 = (mu20 - mu02)**2 + 4*mu11**2
	hu3 = (mu30 - 3*mu12)**2 + (3*mu21 - mu03)**2
	hu4 = (mu30 + mu12)**2 + (mu21 + mu03)**2
	hu5 = (mu30 - 3*mu12)*(mu30 + mu12)*((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) + (3*mu21 - mu03)*(mu21 + mu03)*(3*(mu30 + mu12)**2 - (mu21 + mu03)**2)
	hu6 = (mu20 - mu02)*((mu30 + mu12)**2 - (mu21 + mu03)**2) + 4*mu11*(mu30 + mu12)*(mu21 + mu03)
	hu7 = (3*mu21 - mu03)*(mu30 + mu12)*((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) - (mu30 - 3*mu12)*(mu21 + mu03)*(3*(mu30 + mu12)**2 - (mu21 + mu03)**2)

	# Normalize the moments by dividing them by s^(p+q+2) where p and q are the order of x and y in the moment respectively
	hu1 = hu1 / s**2
	hu2 = hu2 / s**4
	hu3 = hu3 / s**6
	hu4 = hu4 / s**6
	hu5 = hu5 / s**8
	hu6 = hu6 / s**8
	hu7 = hu7 / s**8

	return [hu1, hu2, hu3, hu4, hu5, hu6, hu7]

def calculate_normalized_hu_moments(image):
    """
    Calculate the 7 Hu moments for a 2D image, but normalized as per the original paper!
    """

    if len(image.shape) != 2:
        raise ValueError("Input image must be 2D.")
    
    mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03 = calculate_central_moments(image)
    
    if mu00 == 0:
        raise ValueError("ERROR: Zero area encountered; cannot normalize moments.")
    
    def eta(p, q, mu):
        gamma = (p+q)/2 + 1
        return mu / (mu00 ** gamma)
    
    eta20 = eta(2, 0, mu20)
    eta02 = eta(0, 2, mu02)
    eta11 = eta(1, 1, mu11)
    eta30 = eta(3, 0, mu30)
    eta12 = eta(1, 2, mu12)
    eta21 = eta(2, 1, mu21)
    eta03 = eta(0, 3, mu03)
    
    hu1 = eta20 + eta02
    hu2 = (eta20 - eta02)**2 + 4*(eta11**2)
    hu3 = (eta30 - 3*eta12)**2 + (3*eta21 - eta03)**2
    hu4 = (eta30 + eta12)**2 + (eta21 + eta03)**2
    hu5 = (eta30 - 3*eta12)*(eta30 + eta12)*((eta30 + eta12)**2 - 3*(eta21 + eta03)**2) + (3*eta21 - eta03)*(eta21 + eta03)*(3*(eta30 + eta12)**2 - (eta21 + eta03)**2)
    hu6 = (eta20 - eta02)*((eta30 + eta12)**2 - (eta21 + eta03)**2) + 4*eta11*(eta30 + eta12)*(eta21 + eta03)
    hu7 = (3*eta21 - eta03)*(eta30 + eta12)*((eta30 + eta12)**2 - 3*(eta21 + eta03)**2) - (eta30 - 3*eta12)*(eta21 + eta03)*(3*(eta30 + eta12)**2 - (eta21 + eta03)**2)
    
    return [hu1, hu2, hu3, hu4, hu5, hu6, hu7]

def calculate_geometrically_centered_moments(image, order=3):
    """
    Calculates geometrically centered polynomial moments of a 2D image.

    The moments are computed using spatial coordinates that have been shifted
    by the geometric center (i.e. the mean of the coordinate grid) of the image.
    Each moment is defined as:
    
        M(p, q) = sum_{x,y} [ (x_centered)^p * (y_centered)^q * image(x, y) ]
    
    where the summation is performed over all pixels in the image, and
    (x_centered, y_centered) = (x - mean(x), y - mean(y)).

    The maximum total order of the moments is specified by the parameter `order`.
    
    Note:
        For order = 3, the function returns 10 moments in the following order:
            M(0,0) = sum(image)
            M(1,0) = sum(x_centered * image)
            M(0,1) = sum(y_centered * image)
            M(2,0) = sum(x_centered^2 * image)
            M(1,1) = sum(x_centered * y_centered * image)
            M(0,2) = sum(y_centered^2 * image)
            M(3,0) = sum(x_centered^3 * image)
            M(2,1 = sum(x_centered^2 * y_centered * image)
            M(1,2) = sum(x_centered * y_centered^2 * image)
            M(0,3) = sum(y_centered^3 * image)

    Args:
        image (ndarray): A 2D array representing an image.
        order (int, optional): The maximum order of the polynomial moments to calculate.
                               Must be a non-negative integer (default is 3).

    Returns:
        list: A list of computed polynomial moments in the order specified above.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")

    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Shift the coordinates by subtracting the geometric (grid) mean.
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    moments = []

    for total_order in range(order + 1):
        # Loop over all exponent combinations 
        for j in range(total_order + 1):
            i = total_order - j  # i is the exponent for x_centered, j for y_centered.
            moment = np.sum((x_centered ** i) * (y_centered ** j) * image)
            moments.append(moment)

    return moments

def calculate_fourier_descriptors(image, k=3):
	"""
	Calculates the Fourier Descriptors which are a set of complex 
	numbers that represent the shape of an object, which are calculated 
	by taking the Fourier Transform of the object's boundary. 
	The number of Fourier Descriptors is equal to the number of points 
	on the boundary of the object, as it's calculated by applying the 
	Fourier Transform on the complex representation of the object's boundary.

	The [-k:] notation is used to select the last k elements of the sorted fourier_descriptors array. 
	The sorting is done to put the elements in ascending order, so that the last k elements will be 
	the k largest magnitude Fourier Descriptors. The reason to only keep the k largest magnitude Fourier 
	Descriptors is that they carry the most important shape information of the object, and discarding the 
	rest of the descriptors can simplify the representation of the object without losing much information.

	Note:
		The image is turned to binary, and depending on the image segmentation,
		not all k fourier descriptors may be returned. The default is set to 3 in
		an attempt to include objects with smaller boundaries. 

	Args:
		image (ndarray): A 2D array representing an image.
		k (int): Number of Fourier Descriptors to keep. Defaults to 3.

	Returns:
		The fourier descriptors.
	"""

	if len(image.shape) != 2:
		raise ValueError("Input image must be 2D.")
	if not isinstance(k, int) or k < 0:
		raise ValueError("Order must be a non-negative integer.")

	rows, cols = image.shape
	# Create a grid of x and y coordinates
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))
	# Create binary image
	binary_image = image > 0
	# Find contour of the binary image
	contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	# Select the longest contour
	try:
		contour = max(contours, key=len)
	except:
		print('No contours detected, returning zeros...')
		return [0]*k
	# Convert contour to complex number
	contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
	# Apply Fourier Transform
	fourier_descriptors = np.fft.fft(contour_complex)
	# Keep k largest magnitude Fourier Descriptors
	fourier_descriptors = np.abs(fourier_descriptors)
	fourier_descriptors = [i for i in np.sort(fourier_descriptors)[-k:]]
	if len(fourier_descriptors) != k:
		print('Only {} fourier descriptors could be calculated for this object, returning zeros...'.format(len(fourier_descriptors)))
		for i in range(k-len(fourier_descriptors)):
			fourier_descriptors = fourier_descriptors + [0]

	return fourier_descriptors

	