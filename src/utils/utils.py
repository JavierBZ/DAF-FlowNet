import random
from pathlib import Path

import numpy as np
import torch

import networks
from utils.loss_utils import vector_potential_fn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_model(config, adaptive_fourier_encoding=None):
    architecture = config.network.arch
    if architecture == "FFN":
        return networks.FFN(
            input_dim=config.network.in_dim,
            output_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_dim=config.network.hidden_features,
            fourier_mapping_size=config.network.fourier_mapping_size,
            scale=config.network.fourier_scale,
            adaptive_fourier_encoding=adaptive_fourier_encoding,
        )
    if architecture == "SIREN":
        return networks.SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.first_omega_0,
            hidden_omega_0=config.network.hidden_omega_0,
        )
    raise ValueError(f"Unknown network architecture: {architecture}")


def predict_velocity(model, coordinates, velocity_scale, device, batch_size, use_vector_potential):
    """Predict a native coordinate grid in bounded batches."""
    predictions = []
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        batch = torch.from_numpy(coordinates[start:start + batch_size]).to(device)
        batch.requires_grad_(use_vector_potential)
        with torch.set_grad_enabled(use_vector_potential):
            velocity = model(batch)
            if use_vector_potential:
                velocity = vector_potential_fn(velocity, batch, create_graph=False)
            else:
                velocity = velocity[..., :3]
        predictions.append(velocity.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0) * velocity_scale


def save_checkpoint(path, model, optimizer, iteration, timeframe, training_data, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "iteration": int(iteration),
        "timeframe": int(timeframe),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "velocity_scale": float(training_data.velocity_scale),
        "spacing": tuple(training_data.spacing),
        "dt": float(training_data.dt),
        "shape": tuple(training_data.full_shape),
        "config": config.to_dict(),
    }
    if hasattr(model, "fourier_encoder"):
        checkpoint["fourier_B"] = model.fourier_encoder.B.detach().cpu()
    torch.save(checkpoint, path)
