import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch

from configs.daflownet import get_config
from utils.evaluation_utils import calculate_metrics
from utils.prepare_data import count_timeframes, create_and_normalize_coords, load_frame
from utils.utils import build_model, predict_velocity, resolve_device


def load_model(checkpoint_path, config, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fourier = checkpoint.get("fourier_B")
    encoding = fourier.cpu().numpy() if isinstance(fourier, torch.Tensor) else fourier
    model = build_model(config, adaptive_fourier_encoding=encoding).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def checkpoint_for(output_root, timeframe, explicit=None):
    path = Path(explicit) if explicit is not None else Path(output_root) / f"time_frame_{timeframe}" / "model_final.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def evaluate_timeframe(config, timeframe, checkpoint_path, device, batch_size):
    reference = load_frame(config.data_file_ref, timeframe, config)
    coordinates, _ = create_and_normalize_coords(config, reference.mask.shape, reference.spacing)
    model, checkpoint = load_model(checkpoint_path, config, device)
    velocity_scale = float(checkpoint["velocity_scale"])
    prediction = predict_velocity(
        model,
        coordinates,
        velocity_scale,
        device,
        batch_size,
        config.training.use_vector_potential,
    )
    prediction = prediction.reshape((*reference.mask.shape, 3)).astype(np.float32)
    prediction *= reference.mask[..., None]
    metrics = calculate_metrics(prediction, reference.velocity, reference.mask, reference.spacing)
    metrics["timeframe"] = timeframe
    return prediction, reference, metrics


def write_outputs(output_root, predictions, reference, rows, timeframes, venc):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / "predictions.h5"
    stacked = np.stack(predictions, axis=0)
    with h5py.File(prediction_path, "w") as handle:
        handle.create_dataset("u", data=stacked[..., 0], compression="gzip")
        handle.create_dataset("v", data=stacked[..., 1], compression="gzip")
        handle.create_dataset("w", data=stacked[..., 2], compression="gzip")
        handle.create_dataset("mask", data=reference.mask, compression="gzip")
        handle.create_dataset("timeframes", data=np.asarray(timeframes, dtype=np.int32))
        handle.attrs["spacing"] = reference.spacing
        handle.attrs["dt"] = reference.dt
        handle.attrs["venc"] = venc

    metric_names = ["velocity_nrmse", "direction_error", "divergence_rms"]
    mean_row = {"timeframe": "mean"}
    mean_row.update({name: float(np.mean([row[name] for row in rows])) for name in metric_names})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timeframe", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
    return prediction_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--timeframe", type=int, help="Timeframe to evaluate (default: config t_start).")
    group.add_argument("--all-timeframes", action="store_true", help="Evaluate every configured checkpoint.")
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint for a single timeframe.")
    parser.add_argument("--output-dir", type=Path, help="Model and evaluation directory (default: config output_dir).")
    parser.add_argument("--device", default="auto", help="PyTorch device, such as auto, cpu, or cuda.")
    parser.add_argument("--batch-size", type=int, help="Native-grid prediction batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_timeframes and args.checkpoint is not None:
        raise ValueError("--checkpoint can only be used for one timeframe.")

    config = get_config()
    available = count_timeframes(config.data_file_ref)
    if args.all_timeframes:
        timeframes = list(range(config.domain.t_start, min(config.domain.t_end, available)))
    else:
        timeframes = [config.domain.t_start if args.timeframe is None else args.timeframe]
    if any(timeframe < 0 or timeframe >= available for timeframe in timeframes):
        raise ValueError(f"Timeframe must be between 0 and {available - 1}.")

    output_root = args.output_dir or Path(config.output_dir)
    device = resolve_device(args.device)
    batch_size = args.batch_size or config.evaluation.batch_size
    predictions, rows, last_reference = [], [], None
    for timeframe in timeframes:
        path = checkpoint_for(output_root, timeframe, args.checkpoint)
        prediction, last_reference, metrics = evaluate_timeframe(config, timeframe, path, device, batch_size)
        predictions.append(prediction)
        rows.append(metrics)
        print(
            f"[timeframe {timeframe:03d}] VNRMSE={metrics['velocity_nrmse']:.6g} "
            f"direction={metrics['direction_error']:.6g} divRMS={metrics['divergence_rms']:.6g}"
        )

    if args.all_timeframes:
        evaluation_dir = Path(output_root) / "evaluation"
    else:
        evaluation_dir = Path(output_root) / f"time_frame_{timeframes[0]}" / "evaluation"
    output = write_outputs(evaluation_dir, predictions, last_reference, rows, timeframes, config.constants.venc)
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
