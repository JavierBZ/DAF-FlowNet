
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


def standardize(data, mean=None, std=None):
    mean = np.mean(data) if mean is None else mean
    std = np.std(data) if std is None else std
    return (data - mean) / std, mean, std


def min_max_normalize(data):
    minimum, maximum = np.min(data), np.max(data)
    return (data - minimum) / (maximum - minimum), minimum, maximum


def compute_outer_boundary_mask(mask):
    """Return the six-connected background shell immediately outside ``mask``."""
    fluid = np.asarray(mask, dtype=bool)
    structure = generate_binary_structure(3, 1)
    return (binary_dilation(fluid, structure=structure) & ~fluid).astype(np.uint8)
