
from dataclasses import dataclass

import h5py
import numpy as np

from utils.preprocessing_utils import compute_outer_boundary_mask, min_max_normalize, standardize


@dataclass
class FrameData:
    velocity: np.ndarray
    mask: np.ndarray
    spacing: tuple[float, float, float]
    dt: float


@dataclass
class TrainingData:
    coordinates: np.ndarray
    velocity: np.ndarray
    mask: np.ndarray
    boundary_coordinates: np.ndarray | None
    full_coordinates: np.ndarray
    full_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    dt: float
    velocity_scale: float
    normalization_factors: tuple[float, ...]


def _domain_slices(config):
    return (
        slice(config.domain.x_start, config.domain.x_end),
        slice(config.domain.y_start, config.domain.y_end),
        slice(config.domain.z_start, config.domain.z_end),
    )


def load_frame(path, timeframe, config):
    """Load one velocity frame and its native mask without resampling."""
    spatial = _domain_slices(config)
    with h5py.File(path, "r") as handle:
        velocity = np.stack(
            [np.asarray(handle[name][(timeframe, *spatial)], dtype=np.float32) for name in ("u", "v", "w")],
            axis=-1,
        )
        mask = np.asarray(handle["mask"])
        if mask.ndim == 4:
            mask = mask[0]
        mask = np.asarray(mask[spatial], dtype=np.uint8)
        spacing = tuple(float(value) for value in handle.attrs.get(
            "spacing", (config.resolution.dx, config.resolution.dy, config.resolution.dz)
        ))
        dt = float(handle.attrs.get("dt", config.resolution.dt))
    return FrameData(velocity=velocity, mask=mask, spacing=spacing, dt=dt)


def count_timeframes(path):
    with h5py.File(path, "r") as handle:
        return int(handle["u"].shape[0])


def create_and_normalize_coords(config, shape, spacing):
    """Reproduce the original native-grid coordinate normalization."""
    axes = [np.linspace(step, length * step, length) for length, step in zip(shape, spacing)]

    if config.coords_characteristic:
        axes = [axis / config.constants.L for axis in axes]

    if config.coords_normalization == "standardize":
        if config.global_normalization:
            reference = axes[int(np.argmax([np.ptp(axis) for axis in axes]))]
            global_mean, global_std = float(np.mean(reference)), float(np.std(reference))
            normalized = [standardize(axis, global_mean, global_std)[0] for axis in axes]
            factors = tuple(value for _ in axes for value in (global_mean, global_std))
        else:
            results = [standardize(axis) for axis in axes]
            normalized = [result[0] for result in results]
            factors = tuple(value for result in results for value in result[1:])
    elif config.coords_normalization == "min_max":
        if config.global_normalization:
            minimum = min(float(axis.min()) for axis in axes)
            maximum = max(float(axis.max()) for axis in axes)
            normalized = [(axis - minimum) / (maximum - minimum) for axis in axes]
            factors = tuple(value for _ in axes for value in (minimum, maximum))
        else:
            results = [min_max_normalize(axis) for axis in axes]
            normalized = [result[0] for result in results]
            factors = tuple(value for result in results for value in result[1:])
    else:
        raise ValueError(f"Unknown coordinate normalization: {config.coords_normalization}")

    grids = np.meshgrid(*normalized, indexing="ij")
    coordinates = np.stack([grid.ravel() for grid in grids], axis=1).astype(np.float32)
    return coordinates, factors


def prepare_training_data(config, timeframe):
    frame = load_frame(config.data_file, timeframe, config)
    coordinates, factors = create_and_normalize_coords(config, frame.mask.shape, frame.spacing)

    velocity_scale = float(frame.velocity.max())
    if not np.isfinite(velocity_scale) or velocity_scale <= 0:
        raise ValueError("The maximum velocity must be finite and positive.")
    velocity = (frame.velocity / velocity_scale).reshape(-1, 3).astype(np.float32)

    training_mask = frame.mask.astype(bool)
    if config.setup.expand_mask:
        training_mask |= compute_outer_boundary_mask(training_mask).astype(bool)
    boundary_mask = compute_outer_boundary_mask(training_mask)

    flat_training_mask = training_mask.ravel()
    boundary_coordinates = coordinates[boundary_mask.ravel().astype(bool)] if config.sample_boundary else None
    return TrainingData(
        coordinates=coordinates[flat_training_mask],
        velocity=velocity[flat_training_mask],
        mask=np.ones((int(flat_training_mask.sum()), 1), dtype=np.float32),
        boundary_coordinates=boundary_coordinates,
        full_coordinates=coordinates,
        full_shape=frame.mask.shape,
        spacing=frame.spacing,
        dt=frame.dt,
        velocity_scale=velocity_scale,
        normalization_factors=factors,
    )


def extract_fluid_region(velocity, coordinates, mask):
    fluid = np.asarray(mask).ravel().astype(bool)
    return velocity[fluid], coordinates[fluid]


def sample_collocation_points(config, coordinates, mask):
    """Sample collocation points for retained Vanilla/PINN implementations."""
    count = int(config.collocation_points)
    if config.collocation_in_fluid:
        fluid_coordinates = coordinates[np.asarray(mask).ravel().astype(bool)]
        indices = np.random.choice(len(fluid_coordinates), size=count, replace=True)
        sampled = fluid_coordinates[indices]
        spacings = np.array([
            np.min(np.diff(values)) for values in (np.unique(coordinates[:, axis]) for axis in range(coordinates.shape[1]))
        ])
        return sampled + np.random.uniform(-0.5, 0.5, sampled.shape) * spacings

    minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
    return np.random.uniform(minimum, maximum, size=(count, coordinates.shape[1]))


def sample_boundary_points(coordinates, boundary_mask):
    return coordinates[np.asarray(boundary_mask).ravel().astype(bool)]
