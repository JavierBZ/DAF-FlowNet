import numpy as np


def _masked(values, mask):
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        raise ValueError("The evaluation mask contains no voxels.")
    return selected


def calculate_vnrmse(prediction, reference, mask):
    """Vector RMSE normalized by the maximum reference speed."""
    squared_error = np.sum((prediction - reference) ** 2, axis=-1)
    rmse = np.sqrt(np.mean(_masked(squared_error, mask)))
    maximum_speed = np.max(_masked(np.linalg.norm(reference, axis=-1), mask))
    return float(rmse / maximum_speed) if maximum_speed > 0 else float("nan")


def calculate_directional_error(prediction, reference, mask):
    """Mean 1-cos(theta), matching the manuscript implementation."""
    dot = np.sum(prediction * reference, axis=-1)
    norm_product = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(reference, axis=-1)
    cosine = np.clip(dot / (norm_product + 1e-16), -1.0, 1.0)
    return float(np.mean(_masked(1.0 - cosine, mask)))


def divergence_field(velocity, spacing):
    derivatives = [
        np.gradient(velocity[..., axis], spacing[axis], axis=axis)
        for axis in range(3)
    ]
    return derivatives[0] + derivatives[1] + derivatives[2]


def calculate_divergence_rms(velocity, spacing, mask):
    divergence = divergence_field(velocity, spacing)
    return float(np.sqrt(np.mean(_masked(divergence**2, mask))))


def calculate_metrics(prediction, reference, mask, spacing):
    return {
        "velocity_nrmse": calculate_vnrmse(prediction, reference, mask),
        "direction_error": calculate_directional_error(prediction, reference, mask),
        "divergence_rms": calculate_divergence_rms(prediction, spacing, mask),
    }
