import numpy as np
from astropy.table import Table
from scipy.special import legendre
from math import factorial

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

    # Remove any NaNs and/or infs since even one bad pixel will yield NaN raw moments!
    clean_image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Generate the coordinate grids once since all moments need this
    rows, cols = clean_image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Compute the zeroth image moment it gets passed down when needed
    m00 = np.sum(clean_image)

    if m00 == 0: # In case of completely empty/masked cutouts
        print("[WARNING]: Zero flux image encountered, returning zeros for moments...")
        features = [0.0] * 57
    else: # Compute the moments and pass the pre-computed grids 
        raw_moments = calculate_moments(clean_image, x, y, m00)
        central_moments = calculate_central_moments(clean_image, x, y, m00)
        geo_moments = calculate_geometrically_centered_moments(clean_image, x, y)
        hu_moments = calculate_hu_moments(clean_image, central_moments=central_moments)
        zernike_moments = calculate_zernike_moments(clean_image, x=x, y=y)

        if rows != cols:
            legendre_moments = [np.nan] * 10
            print('[WARNING]: Legendre moments require square input. Consider resizing or cropping. Returning NaN Legendre moments')
        else:
            legendre_moments = calculate_legendre_moments(clean_image)

        features = raw_moments + central_moments + geo_moments + hu_moments + legendre_moments + zernike_moments

    col_names = [
        'M00', 'M10', 'M01', 'M20', 'M11', 'M02', 'M30', 'M21', 'M12', 'M03',
        'mu00', 'mu10', 'mu01', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
        'G00', 'G10', 'G01', 'G20', 'G11', 'G02', 'G30', 'G21', 'G12', 'G03',
        'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7',
        'L00', 'L10', 'L01', 'L20', 'L11', 'L02', 'L30', 'L21', 'L12', 'L03',
        'Z0_0', 'Z1_m1', 'Z1_1', 'Z2_m2', 'Z2_0', 'Z2_2', 'Z3_m3', 'Z3_m1', 'Z3_1', 'Z3_3'
    ]

    dtype = ('f8',) * len(col_names)
    return Table(data=np.array([features]), names=np.array(col_names), dtype=dtype)

def calculate_moments(image: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, m00: float = None) -> list:
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

    if x is None or y is None:
        rows, cols = image.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    if m00 is None:
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

def calculate_central_moments(image: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, m00: float = None) -> list:
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

    if x is None or y is None:
        rows, cols = image.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    if m00 is None:
        m00 = np.sum(image)

    if m00 == 0:
        print('[WARNING]: Zeroth image moment is zero! Raw image moments are set to zero...')
        return [0.0] * 10

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

def calculate_geometrically_centered_moments(image: np.ndarray, max_order: int = 3, x: np.ndarray = None, y: np.ndarray = None) -> list:
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

    if x is None or y is None:
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

def zernike_radial_polynomial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """
    Compute the Zernike radial polynomial R_n^m(r).
    """

    m = abs(m)
    if (n - m) % 2 != 0 or n < m:
        return np.zeros_like(r)
    
    R = np.zeros_like(r)
    for s in range((n - m) // 2 + 1):
        numerator = (-1)**s * factorial(n - s)
        denominator = factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)
        R += (numerator / denominator) * (r ** (n - 2 * s))

    return R

def calculate_zernike_moments(image: np.ndarray, max_order: int = 3, x: np.ndarray = None, y: np.ndarray = None) -> list:
    """
    Compute the magnitudes of Zernike moments up to a given order.
    Zernike moments are evaluated on a unit disk and their magnitudes are rotationally invariant.

    Parameters
    ----------
    image : ndarray
        2D array representing a greyscale image.
    max_order : int, optional
        Maximum radial order (n) of the moments (default is 3).
    x, y : ndarray, optional
        Precomputed coordinate grids.

    Returns
    -------
    list of float
        Magnitudes of Zernike moments |Z_nm| for valid (n, m) pairs where n <= max_order.
    """

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    rows, cols = image.shape
    if x is None or y is None:
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Center coordinates and normalize to a unit disk
    x_c = x - (cols - 1) / 2.0
    y_c = y - (rows - 1) / 2.0
    
    # Calc polar coordinates
    r = np.sqrt(x_c**2 + y_c**2)
    theta = np.arctan2(y_c, x_c)
    
    # Normalize so the max radius in the image bounds fits in the unit disk
    r_max = np.max(r)
    if r_max > 0:
        r = r / r_max

    # Only need to evaluate pixels within r <= 1, should be ok since normalized above but masking just in case
    mask = r <= 1.0

    zernike_magnitudes = []
    
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:  # Zernike polynomials are only defined for (n - |m|) = even
                R = zernike_radial_polynomial(n, m, r[mask])

                # The Zernike basis function
                V = R * np.exp(-1j * m * theta[mask]) # V_nm = R_n^m * exp(-j * m * theta)
                
                Z = ((n + 1) / np.pi) * np.sum(image[mask] * np.conj(V)) # Z_nm = (n + 1) / pi * sum(image * V*)
                
                # We append the magnitude of the complex moment for rotational invariance
                zernike_magnitudes.append(float(np.abs(Z)))

    return zernike_magnitudes




