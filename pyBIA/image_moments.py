from astropy.table import Table
import numpy as np
import cv2

def make_moments_table(image):
	"""
	This function takes a 2D image array as input and 
	calculates the image moments, central moments, Hu moments, 
	and legendre moments to third order. The fourier descriptors
	are also computed but only the first ten are kept.
	These 47 features are concatenated and then saved in an astropy Table.

	Args:
		image (ndarray): A 2D array representing an image.

	Returns:
		A astropy table with 47 columns, one for each moment or descriptor and their corresponding values as rows."
    """

	moments, central_moments, hu_moments = calculate_moments(image), calculate_central_moments(image), calculate_hu_moments(image)
	legendre_moments, fourier_descriptors = calculate_legendre_moments(image), calculate_fourier_descriptors(image, k=3)
	features = moments + central_moments + hu_moments + fourier_descriptors + legendre_moments

	col_names = ['m00','m10','m01','m20','m11','m02','m30','m21','m12','m03',
	             'mu00','mu10','mu01','mu20','mu11','mu02','mu30','mu21','mu12','mu03',
	             'hu1','hu2','hu3','hu4','hu5','hu6','hu7',
	             'fourier_1','fourier_2','fourier_3',
	             'legendre_1','legendre_2','legendre_3','legendre_4','legendre_5','legendre_6','legendre_7','legendre_8','legendre_9','legendre_10']

	features, col_names = np.array(features), np.array(col_names)
	dtype = ('f8',) * len(col_names)
	features_table = Table(data=features, names=col_names, dtype=dtype)
	
	return features_table

def calculate_moments(image):
	"""
	This function takes a 2D image array as input 
	and calculates the image moments to third order.

	Args:
		image (ndarray): A 2D array representing an image.

	Returns: 
		A tuple of 10 values representing the calculated image moments (m00, m10, m01, m20, m11, m02, m30, m21, m12, m03)
	"""

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

	rows, cols = image.shape
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))

	m00 = np.sum(image)
	x_bar = np.sum(x * image) / m00
	y_bar = np.sum(y * image) / m00

	mu00 = m00
	mu10 = 0
	mu01 = 0
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

def calculate_legendre_moments(image, order=3):
	"""
	This function takes a 2D image array calculates 
	the Legendre moments.

	If the order is set to 3, there would be 10 Legendre moments in total. 
	They would be represented by the monomials x^3, y^3, x^2y, xy^2, x^2, y^2, xy, x, y, 1.

	Args:
		image (ndarray): A 2D array representing an image.
		order (int): The order of the Legendre moments to calculate.

	Returns:
		A list of Legendre moments.
	"""

	x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
	x = x - np.mean(x)
	y = y - np.mean(y)
	moments = []
	for i in range(order+1):
		for j in range(i+1):
			moment = np.sum(np.power(x, i-j) * np.power(y, j) * image)
			moments.append(moment)
	
	return moments

def calculate_fourier_descriptors(image, k=10):
	"""
	Calculates the Fourier Descriptors which are a set of complex 
	numbers that represent the shape of an object, which are calculated 
	by taking the Fourier Transform of the object's boundary. 
	The number of Fourier Descriptors is equal to the number of points 
	on the boundary of the object, as it's calculated by applying the 
	Fourier Transform on the complex representation of the object's boundary.

	Args:
		image (ndarray): A 2D array representing an image.
		k (int): Number of Fourier Descriptors to keep. Defaults to 3.

	Returns:
		The fourier descriptors.
	"""

	rows, cols = image.shape
	# Create a grid of x and y coordinates
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))
	# Create binary image
	binary_image = image > 0
	# Find contour of the binary image
	contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	# Select the longest contour
	contour = max(contours, key=len)
	# Convert contour to complex number
	contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
	# Apply Fourier Transform
	fourier_descriptors = np.fft.fft(contour_complex)
	# Keep k largest magnitude Fourier Descriptors
	fourier_descriptors = np.abs(fourier_descriptors)
	fourier_descriptors = [i for i in np.sort(fourier_descriptors)[-k:]]
	return fourier_descriptors

def zernike_moments(image, r_max=3):
	"""
	This function takes a 2D image array and 
	an integer "r_max" as input and calculates 
	the Zernike moments up to the order of "r_max".

	If r_max=3, there are 13 Zernike moments. Each Zernike moment 
	is indexed by two integers n and m, where n is the radial order 
	and m is the azimuthal order. The radial order must be non-negative 
	and the azimuthal order must be in the range -n <= m <= n. The total 
	number of Zernike moments is given by the formula (n+1)^2. Therefore, 
	if r_max is 3, the highest radial order is 3, and the total number of 
	Zernike moments is (3+1)^2 = 4^2 = 16. However, since the Zernike moment 
	Z_{n, m} is equivalent to Z_{n, -m}, only half of them are unique. 
	Therefore, the number of unique Zernike moments is 16/2 = 8.

	Args:
		image (ndarray): A 2D array representing an image.
		r_max (int): The maximum order of the Zernike moments to calculate.

	Returns:
		A tuple of (r_max+1)*(r_max+1) values representing the calculated Zernike moments.
	"""

	rows, cols = image.shape
	# Create a grid of x and y coordinates
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))
	x = x - np.mean(x)
	y = y - np.mean(y)

	# Convert x and y coordinates to polar coordinates
	rho = np.sqrt(x**2 + y**2)
	theta = np.arctan2(y, x)

	moments = []
	# Define the range of n and m values
	n_range = range(r_max + 1)
	m_range = range(-r_max, r_max + 1, 2)
	# Calculate the Zernike moments
	for n in n_range:
		for m in m_range:
			if n < m:
				continue
			Rnm = zernike_radial(n, m, rho)
			Znm = Rnm * np.exp(1j * m * theta)
			moment = np.sum(image * Znm)
			moments.append(moment)
	return moments

def harris_corner_descriptors(image, block_size=2, ksize=3):
	"""
	Harris Corner Descriptors are calculated by applying 
	the Harris corner detection algorithm on the grayscale image. 
	The Harris corner detection algorithm uses the second derivatives 
	of the image intensity to detect corners in an image. It's sensitive 
	to both the intensity and the location of the corner.

	The Harris Corner Descriptors are returned as a 2D array, 
	where each element represents the corner response at that particular pixel. 
	The higher the value of a pixel, the more likely it is to be a corner.

	Args:
		image (ndarray): A 2D array representing an image.
		block_size (int): The block_size parameter is used to set the size of 
			the neighborhood used in the calculation of the Harris corner response. 
			A larger block size will result in averaging over a larger neighborhood 
			and may be less sensitive to noise. Defaults to 2.
		ksize (int): This parameter is used to set the sensitivity of the Harris corner 
			detector. A larger value of ksize will result in a higher threshold for detecting corners.
			Defaults to 3.

	Returns:
		The Harris corner descriptors.
	"""

	# Convert image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Apply Harris corner detection
	corners = cv2.cornerHarris(gray, block_size, ksize, 0.04)
	# Normalize corner responses
	corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	return corners



def zernike_radial(n, m, rho):
	"""
	This function takes integers n and m 
	and a float rho as input and returns 
	the radial polynomial Zernike moment of order (n, m)

	Args:
		n (int): The n order of Zernike polynomial.
		m (int): The m order of Zernike polynomial.
		rho (float): A value between 0 and 1 representing the distance from the center of the image.

	Returns:
		The radial polynomial Zernike moment of order (n, m)
	"""

	if (n - m) % 2 != 0:
	    return 0
	k = (n - m) // 2
	pre_factor = (-1) ** k * comb(n, k)
	Rnm = pre_factor * rho ** m
	for i in range(1, k + 1):
		pre_factor = -pre_factor * (n - i) / (i * (2 * i + m))
		Rnm = Rnm * (rho ** 2 - i)
	return Rnm

def comb(n, k):
	"""
	This function takes integers n and k as 
	input and returns the value of the binomial 
	coefficient of n and k.

	Args:
		n (int): The number of elements in the set
		k (int): The number of elements to choose

	Returns:
		The binomial coefficient of n and k
	"""

	return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))



