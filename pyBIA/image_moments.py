import numpy as np
from astropy.table import Table
from scipy.special import legendre


def make_moments_table(image: np.ndarray) -> Table:
    """
    Compute a full set of image moments and return as an Astropy Table.

    This function computes raw moments, central moments, Hu moments,
    geometrically centered polynomial moments, and Legendre moments.
    It is the main feature-generation function used in pyBIA to build morphology catalogs.

    Parameters
    ----------
    image : ndarray
        2D array representing a greyscale image.

    Returns
    -------
    astropy.table.Table
        A table with 47 features including raw, central, geometrically centered, Hu,
        and Legendre moments.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if image.shape[0] != image.shape[1]:
        raise ValueError("Legendre moments require square input. Consider resizing or cropping.")


    raw_moments = calculate_moments(image)
    central_moments = calculate_central_moments(image)
    geo_moments = calculate_geometrically_centered_moments(image)
    hu_moments = calculate_hu_moments(image, central_moments=central_moments)
    legendre_moments = calculate_legendre_moments(image)

    features = raw_moments + central_moments + geo_moments + hu_moments + legendre_moments

    col_names = [
        'M00', 'M10', 'M01', 'M20', 'M11', 'M02', 'M30', 'M21', 'M12', 'M03',
        'mu00', 'mu10', 'mu01', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
        'G00', 'G10', 'G01', 'G20', 'G11', 'G02', 'G30', 'G21', 'G12', 'G03',
        'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7',
        'L00', 'L10', 'L01', 'L20', 'L11', 'L02', 'L30', 'L21', 'L12', 'L03'
    ]

    dtype = ('f8',) * len(col_names)
    return Table(data=np.array(features), names=np.array(col_names), dtype=dtype)

def calculate_moments(image: np.ndarray) -> list:
    """
    Compute raw spatial moments up to 3rd order.

    Parameters
    ----------
    image : ndarray
        2D array representing a greyscale image.

    Returns
    -------
    list of float
        Raw moments [m00, m10, m01, ..., m03].
    """
    if image.ndim != 2:
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

def calculate_central_moments(image: np.ndarray) -> list:
    """
    Compute central moments up to 3rd order.

    Parameters
    ----------
    image : ndarray
        2D array representing a greyscale image.

    Returns
    -------
    list of float
        Central moments [mu00, mu10, ..., mu03].
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    m00 = np.sum(image)
    x_bar = np.sum(x * image) / m00
    y_bar = np.sum(y * image) / m00

    mu00 = m00
    mu10 = np.sum((x - x_bar) * image) #Should be 0 but usually get ~1e-15 - 1e-16
    mu01 = np.sum((y - y_bar) * image) #Should be 0 but usually get ~1e-15 - 1e-16
    mu20 = np.sum((x - x_bar)**2 * image)
    mu11 = np.sum((x - x_bar) * (y - y_bar) * image)
    mu02 = np.sum((y - y_bar)**2 * image)
    mu30 = np.sum((x - x_bar)**3 * image)
    mu21 = np.sum((x - x_bar)**2 * (y - y_bar) * image)
    mu12 = np.sum((x - x_bar) * (y - y_bar)**2 * image)
    mu03 = np.sum((y - y_bar)**3 * image)

    return [mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03]

def calculate_geometrically_centered_moments(image: np.ndarray, max_order: int = 3) -> list:
    """
    Compute raw polynomial moments centered on the geometric center of the image grid.

    Parameters
    ----------
    image : ndarray
        2D array representing a greyscale image.
    max_order : int, optional
        Maximum total order of the moments (default is 3).

    Returns
    -------
    list of float
        Polynomial moments of the form sum(x^p * y^q * image(x,y)) for p+q ≤ max_order,
        where (x, y) are pixel coordinates shifted so that (0,0) is the geometric center
        of the image.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if not isinstance(max_order, int) or max_order < 0:
        raise ValueError("Order must be a non-negative integer.")

    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Shift pixel coordinates to geometric center of image grid
    x_centered = x - (cols - 1) / 2
    y_centered = y - (rows - 1) / 2

    moments = []

    for total_order in range(max_order + 1):
        for j in range(total_order + 1):
            i = total_order - j
            moment = np.sum((x_centered ** i) * (y_centered ** j) * image)
            moments.append(moment)

    return moments

def calculate_hu_moments(image: np.ndarray, central_moments=None) -> list:
    """
    Compute Hu moments using classical eta-normalization.

    Parameters
    ----------
    image : ndarray
        2D array representing a greyscale image.
    central_moments : list of float, optional
        Precomputed central moments to third order (10 total: [mu00, mu10, ..., mu03]). If not provided, they will be computed from the image.

    Returns
    -------
    list of float
        Classical Hu moments [hu1, ..., hu7] using eta(p,q) invariants.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    if central_moments is None:
    	mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03 = calculate_central_moments(image)
    else:
    	mu00, mu10, mu01, mu20, mu11, mu02, mu30, mu21, mu12, mu03 = central_moments

    if mu00 == 0:
        #raise ValueError("ERROR: Zero area encountered; cannot normalize moments.")
        print("ERROR: Zero area encountered; cannot normalize moments. Returning NaN...")
        return [np.nan] * 7
    
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

def calculate_legendre_moments(image: np.ndarray, max_order: int = 3) -> list[float]:
    """
    Compute 2D Legendre moments L_nm for n + m ≤ max_order using orthonormal basis.

    Parameters
    ----------
    image : ndarray
        2D array representing a greyscale image.
    max_order : int
        Maximum total order (n + m) of Legendre moments to compute.

    Returns
    -------
    list of float
        Legendre moments up to n+m ≤ max_order.
    """
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Input must be a square 2‑D array")

    N = image.shape[0]
    coords = (2 * np.arange(N) - N + 1) / (N - 1)

    # Legendre polynomials using scipy
    polynomials = [legendre(k)(coords) for k in range(max_order + 1)]

    moments = []
    norm = 1.0 / (N - 1) ** 2

    for n in range(max_order + 1):
        Pn = polynomials[n][:, None]
        for m in range(max_order + 1 - n):
            Pm = polynomials[m][None, :]
            kernel = Pn * Pm
            L_nm = (2 * n + 1) * (2 * m + 1) * norm * np.sum(image * kernel)
            moments.append(float(L_nm))

    return moments

