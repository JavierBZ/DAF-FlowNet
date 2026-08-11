import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from configs.daflownet import get_config
from utils.loss_utils import compute_boundary_loss, compute_data_loss
from utils.prepare_data import count_timeframes, prepare_training_data
from utils.training_outputs import write_training_outputs
from utils.utils import build_model, resolve_device, save_checkpoint, set_seed


def _sample(array, count):
    if count is None or count >= len(array):
        return array
    return array[np.random.choice(len(array), size=count, replace=False)]


def _wandb_run(enabled, project, entity, name, config):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb or run without --wandb.") from error
    return wandb.init(project=project, entity=entity, name=name, config=config.to_dict())


def train_timeframe(config, timeframe, output_root, device, wandb_options=None):
    set_seed(config.random_seed)
    data = prepare_training_data(config, timeframe)
    config.U_max = data.velocity_scale

    frame_dir = Path(output_root) / f"time_frame_{timeframe}"
    checkpoint_dir = frame_dir / "checkpoints"
    frame_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    run_name = f"{config.network_name}_time_frame_{timeframe}"
    options = wandb_options or {}
    run = _wandb_run(options.get("enabled", False), options.get("project"), options.get("entity"), options.get("name") or run_name, config)

    fieldnames = ["iteration", "total_loss", "data_loss", "boundary_loss", "learning_rate", "seconds"]
    log_path = frame_dir / "training.csv"
    start = time.time()
    try:
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()

            for iteration in range(1, config.training.iterations + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)

                count = config.training.data_points_per_batch
                if count is None or count >= len(data.coordinates):
                    indices = np.arange(len(data.coordinates))
                else:
                    indices = np.random.choice(len(data.coordinates), size=count, replace=False)

                coordinates = torch.from_numpy(data.coordinates[indices]).to(device).requires_grad_(True)
                velocity = torch.from_numpy(data.velocity[indices]).to(device)
                mask = torch.from_numpy(data.mask[indices]).to(device)
                data_loss, _ = compute_data_loss(config, model, coordinates, velocity, mask)

                boundary_loss = torch.zeros((), device=device)
                if config.sample_boundary:
                    boundary = _sample(data.boundary_coordinates, config.training.boundary_points_per_batch)
                    boundary = torch.from_numpy(boundary).to(device)
                    boundary_loss = compute_boundary_loss(config, model, boundary)

                total_loss = data_loss + config.training.boundary_weight * boundary_loss
                total_loss.backward()
                optimizer.step()

                if iteration % config.training.lr_decay_iter == 0:
                    for group in optimizer.param_groups:
                        group["lr"] *= config.training.lr_decay_factor

                should_log = iteration == 1 or iteration % config.training.log_iter == 0 or iteration == config.training.iterations
                if should_log:
                    row = {
                        "iteration": iteration,
                        "total_loss": float(total_loss.detach()),
                        "data_loss": float(data_loss.detach()),
                        "boundary_loss": float(boundary_loss.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "seconds": time.time() - start,
                    }
                    writer.writerow(row)
                    stream.flush()
                    print(
                        f"[timeframe {timeframe:03d}, iteration {iteration:04d}] "
                        f"loss={row['total_loss']:.6g} data={row['data_loss']:.6g} "
                        f"boundary={row['boundary_loss']:.6g}"
                    )
                    if run is not None:
                        run.log({key: value for key, value in row.items() if key not in {"iteration", "seconds"}}, step=iteration)

                if iteration % config.training.summary_iter == 0:
                    save_checkpoint(checkpoint_dir / f"model_iteration_{iteration:04d}.pth", model, optimizer, iteration, timeframe, data, config)

                write_errors = iteration == 1 or iteration % config.training.error_iter == 0 or iteration == config.training.iterations
                write_plot = iteration % config.plot.iter == 0 or iteration == config.training.iterations
                if write_errors or write_plot:
                    metrics = write_training_outputs(
                        config, model, device, iteration, timeframe, frame_dir, data, write_errors, write_plot
                    )
                    if metrics is not None:
                        print(
                            f"[timeframe {timeframe:03d}, iteration {iteration:04d}] "
                            f"VNRMSE={metrics['velocity_nrmse']:.6g} "
                            f"direction={metrics['direction_error']:.6g} "
                            f"divRMS={metrics['divergence_rms']:.6g}"
                        )
                        if run is not None:
                            run.log(metrics, step=iteration)

        final_path = frame_dir / "model_final.pth"
        save_checkpoint(final_path, model, optimizer, config.training.iterations, timeframe, data, config)
        return final_path
    finally:
        if run is not None:
            run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--timeframe", type=int, help="Timeframe to train (default: config t_start).")
    group.add_argument("--all-timeframes", action="store_true", help="Train every configured timeframe independently.")
    parser.add_argument("--iterations", type=int, help="Override the configured iteration count, useful for smoke tests.")
    parser.add_argument("--output-dir", type=Path, help="Checkpoint and log directory (default: config output_dir).")
    parser.add_argument("--device", default="auto", help="PyTorch device, such as auto, cpu, or cuda.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="DAF-FlowNet")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    if args.iterations is not None:
        if args.iterations < 1:
            raise ValueError("--iterations must be positive.")
        config.training.iterations = args.iterations

    available = count_timeframes(config.data_file)
    if args.all_timeframes:
        timeframes = range(config.domain.t_start, min(config.domain.t_end, available))
    else:
        timeframes = [config.domain.t_start if args.timeframe is None else args.timeframe]
    if any(timeframe < 0 or timeframe >= available for timeframe in timeframes):
        raise ValueError(f"Timeframe must be between 0 and {available - 1}.")

    device = resolve_device(args.device)
    output_root = args.output_dir or Path(config.output_dir)
    wandb_options = {
        "enabled": args.wandb,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
    }
    print(f"Training on {device}; outputs: {output_root}")
    for timeframe in timeframes:
        train_timeframe(config, timeframe, output_root, device, wandb_options)


if __name__ == "__main__":
    main()
