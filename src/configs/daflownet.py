"""DAF-FlowNet configuration for the in-silico example."""

from pathlib import Path

import ml_collections


def get_config():
    root = Path(__file__).resolve().parents[2]
    config = ml_collections.ConfigDict()

    config.data_file = str(root / "data" / "sim.h5")
    config.data_file_ref = str(root / "data" / "ref.h5")
    config.output_dir = str(root / "results" / "model1")
    config.network_name = "DAF_FlowNet"
    config.random_seed = 123

    config.domain = ml_collections.ConfigDict()
    config.domain.t_start = 3
    config.domain.t_end = 29
    config.domain.x_start = 0
    config.domain.x_end = 60
    config.domain.y_start = 0
    config.domain.y_end = 30
    config.domain.z_start = 0
    config.domain.z_end = 38

    config.resolution = ml_collections.ConfigDict()
    config.resolution.from_file = True
    config.resolution.dx = 0.002
    config.resolution.dy = 0.002
    config.resolution.dz = 0.002
    config.resolution.dt = 0.01874

    config.setup = ml_collections.ConfigDict()
    config.setup.fluid_region = True
    config.setup.expand_mask = True
    config.setup.include_time = False
    config.setup.include_pressure = False

    config.sample_collocation = False
    config.collocation_in_fluid = True
    config.collocation_points = 50_000
    config.sample_boundary = True

    config.global_normalization = True
    config.coords_characteristic = False
    config.coords_normalization = "min_max"
    config.vel_normalization = "max_velocity"

    config.constants = ml_collections.ConfigDict()
    config.constants.U = 1.0
    config.constants.L = 0.005
    config.constants.T = 0.005
    config.constants.rho = 1060
    config.constants.mu = 0.004
    config.constants.venc = 1.5

    config.network = ml_collections.ConfigDict()
    config.network.arch = "FFN"
    config.network.in_dim = 3
    config.network.out_dim = 3
    config.network.depth = 5
    config.network.hidden_features = 200
    config.network.fourier_mapping_size = 256
    config.network.fourier_scale = 1.0
    # Retained for the comparison network implementations.
    config.network.first_omega_0 = 30
    config.network.hidden_omega_0 = 30

    config.training = ml_collections.ConfigDict()
    config.training.iterations = 3_500
    config.training.data_points_per_batch = None
    config.training.boundary_points_per_batch = None
    config.training.lr = 1e-3
    config.training.lr_decay_iter = 500
    config.training.lr_decay_factor = 0.8
    config.training.log_iter = 25
    config.training.summary_iter = 1_000
    config.training.error_iter = 500

    config.training.use_mse = False
    config.training.use_cosine = True
    config.training.use_vector_potential = True
    config.training.u_weight = 1.0
    config.training.v_weight = 1.0
    config.training.w_weight = 1.0
    config.training.p_weight = 0.01
    config.training.pressure_in_data_loss = False

    config.training.use_physics_loss = False
    config.training.physics_loss_on_data_points = False
    config.training.use_navier_stokes = False
    config.training.use_divergence = False
    config.training.physics_weight = 1.0
    config.training.epochs_before_PDE = 0

    config.training.use_boundary_mse = True
    config.training.pressure_in_boundary_loss = False
    config.training.boundary_weight = 1.0

    config.evaluation = ml_collections.ConfigDict()
    config.evaluation.batch_size = 16_384

    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 1_000
    config.plot.slice_axis = "y"
    config.plot.slice_index = 15

    return config
