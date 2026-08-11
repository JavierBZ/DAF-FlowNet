
import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.evaluation_utils import calculate_metrics
from utils.prepare_data import load_frame
from utils.utils import predict_velocity


def _slice(field, axis, index):
    dimension = {"x": 0, "y": 1, "z": 2}[axis]
    index = min(int(index), field.shape[dimension] - 1)
    return np.take(field, index, axis=dimension)


def _save_prediction_plot(path, prediction, mask, slice_axis, slice_index):
    masked = prediction * mask[..., None]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for component, axis, name in zip(range(3), axes, ("u", "v", "w")):
        image = axis.imshow(_slice(masked[..., component], slice_axis, slice_index), origin="lower", cmap="viridis")
        axis.set_title(f"Predicted {name}")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_error_plots(directory, noisy, reference, prediction, mask, slice_axis, slice_index):
    for component, name in enumerate(("u", "v", "w")):
        fields = (
            noisy[..., component] * mask,
            reference[..., component] * mask,
            prediction[..., component] * mask,
            np.abs(reference[..., component] - prediction[..., component]) * mask,
        )
        titles = (f"Noisy {name}", f"Reference {name}", f"Predicted {name}", f"Absolute error {name}")
        figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        value_min = min(float(fields[1].min()), float(fields[2].min()))
        value_max = max(float(fields[1].max()), float(fields[2].max()))
        for index, (axis, field, title) in enumerate(zip(axes, fields, titles)):
            limits = {} if index == 3 else {"vmin": value_min, "vmax": value_max}
            image = axis.imshow(_slice(field, slice_axis, slice_index), origin="lower", cmap="viridis", **limits)
            axis.set_title(title)
            figure.colorbar(image, ax=axis, shrink=0.75)
        figure.savefig(directory / f"prediction_vs_reference_{name}.png", dpi=150)
        plt.close(figure)


def write_training_outputs(config, model, device, iteration, timeframe, frame_dir, training_data, write_errors, write_plot):
    prediction = predict_velocity(
        model,
        training_data.full_coordinates,
        training_data.velocity_scale,
        device,
        config.evaluation.batch_size,
        config.training.use_vector_potential,
    ).reshape((*training_data.full_shape, 3)).astype(np.float32)
    noisy = load_frame(config.data_file, timeframe, config)
    reference = load_frame(config.data_file_ref, timeframe, config)
    prediction *= reference.mask[..., None]
    slice_axis = config.plot.slice_axis
    slice_index = config.plot.slice_index

    metrics = None
    if write_errors:
        error_dir = Path(frame_dir) / "errors" / f"iter_{iteration}"
        error_dir.mkdir(parents=True, exist_ok=True)
        metrics = calculate_metrics(prediction, reference.velocity, reference.mask, reference.spacing)
        with (error_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        with h5py.File(error_dir / "prediction.h5", "w") as handle:
            for component, name in enumerate(("u", "v", "w")):
                handle.create_dataset(name, data=prediction[..., component], compression="gzip")
            handle.create_dataset("mask", data=reference.mask, compression="gzip")
            handle.attrs["spacing"] = reference.spacing
            handle.attrs["dt"] = reference.dt
            handle.attrs["venc"] = config.constants.venc
        _save_error_plots(
            error_dir, noisy.velocity, reference.velocity, prediction, reference.mask, slice_axis, slice_index
        )

    if write_plot:
        plot_dir = Path(frame_dir) / "plots" / f"iter_{iteration}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_prediction_plot(plot_dir / "predictions.png", prediction, reference.mask, slice_axis, slice_index)

    return metrics
