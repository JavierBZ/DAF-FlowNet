import numpy as np
import torch

def mse_loss(uvw_data, uvw_pred, config):
    
    u_pred, v_pred, w_pred = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2]
    u, v, w = uvw_data[..., 0], uvw_data[..., 1], uvw_data[..., 2]

    mse_u = (u_pred - u) ** 2
    mse_v = (v_pred - v) ** 2
    mse_w = (w_pred - w) ** 2

    mse_vel = torch.mean(config.training.u_weight*mse_u + config.training.v_weight*mse_v + config.training.w_weight*mse_w)

    if config.training.pressure_in_data_loss:
        p_pred = uvw_pred[..., 3]
        p = uvw_data[..., 3]

        mse_p = (p_pred - p) ** 2

        mse_vel += config.training.p_weight*torch.mean(mse_p)

    return mse_vel

def cosine_loss(uvw_data, uvw_pred, config):
    
    u_pred, v_pred, w_pred = uvw_pred[..., 0]*config.U_max, uvw_pred[..., 1]*config.U_max, uvw_pred[..., 2]*config.U_max
    u, v, w = uvw_data[..., 0]*config.U_max, uvw_data[..., 1]*config.U_max, uvw_data[..., 2]*config.U_max

    Kv = torch.pi / config.constants.venc
    cosine_u = 1 - torch.cos(Kv * (u_pred - u))
    cosine_v = 1 - torch.cos(Kv * (v_pred - v))
    cosine_w = 1 - torch.cos(Kv * (w_pred - w))

    cosine_vel = torch.mean(config.training.u_weight*cosine_u + config.training.v_weight*cosine_v + config.training.w_weight*cosine_w)

    if config.training.pressure_in_data_loss:
        p_pred = uvw_pred[..., 3]
        p = uvw_data[..., 3]

        cosine_p = 1 - torch.cos(Kv * (p_pred - p))
        
        cosine_vel += config.training.p_weight*torch.mean(cosine_p)
    
    return cosine_vel

def fluid_weighted_mse_loss(uvw_data, uvw_pred, mask, config):
    
    u_pred, v_pred, w_pred = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2]
    u, v, w = uvw_data[..., 0], uvw_data[..., 1], uvw_data[..., 2]

    mse_u = (u_pred - u) ** 2
    mse_v = (v_pred - v) ** 2
    mse_w = (w_pred - w) ** 2

    # Get non-fluid mask
    mask_flattened = mask.flatten()
    non_fluid_mask = 1 - mask_flattened

    mse_u_fluid = mse_u * mask_flattened
    mse_u_non_fluid = mse_u * non_fluid_mask

    mse_v_fluid = mse_v * mask_flattened
    mse_v_non_fluid = mse_v * non_fluid_mask

    mse_w_fluid = mse_w * mask_flattened
    mse_w_non_fluid = mse_w * non_fluid_mask

    mse_u_total = config.training.u_weight*(mse_u_fluid.sum()/mask_flattened.sum() + mse_u_non_fluid.sum()/non_fluid_mask.sum())
    mse_v_total = config.training.v_weight*(mse_v_fluid.sum()/mask_flattened.sum() + mse_v_non_fluid.sum()/non_fluid_mask.sum())
    mse_w_total = config.training.w_weight*(mse_w_fluid.sum()/mask_flattened.sum() + mse_w_non_fluid.sum()/non_fluid_mask.sum())

    mse_vel = torch.mean(mse_u_total + mse_v_total + mse_w_total)

    if config.training.pressure_in_data_loss:

        p_pred = uvw_pred[..., 3]
        p = uvw_data[..., 3]

        mse_p = (p_pred - p) ** 2
        mse_p_fluid = mse_p * mask
        mse_p_non_fluid = mse_p * non_fluid_mask

        mse_p = mse_p_fluid.sum()/mask.sum() + mse_p_non_fluid.sum()/non_fluid_mask.sum()
        mse_vel += config.training.p_weight*mse_p

    return mse_vel

def vector_potential_fn(phi_pred, xyz_hr, create_graph=True):

   # Separate out each predicted component
    phi_x = phi_pred[:, 0]  # shape (N,)
    phi_y = phi_pred[:, 1]
    phi_z = phi_pred[:, 2]

    # Compute partial derivatives
    dphix = torch.autograd.grad(phi_x, xyz_hr,
                                grad_outputs=torch.ones_like(phi_x),
                                create_graph=create_graph, retain_graph=True)[0]  # shape (N,3)

    dphiy = torch.autograd.grad(phi_y, xyz_hr,
                                grad_outputs=torch.ones_like(phi_y),
                                create_graph=create_graph, retain_graph=True)[0]

    dphiz = torch.autograd.grad(phi_z, xyz_hr,
                                grad_outputs=torch.ones_like(phi_z),
                                create_graph=create_graph, retain_graph=True)[0]

    # Extract partial derivatives for each dimension:
    _, dphix_dy, dphix_dz = dphix[:, 0], dphix[:, 1], dphix[:, 2]
    dphiy_dx, _, dphiy_dz = dphiy[:, 0], dphiy[:, 1], dphiy[:, 2]
    dphiz_dx, dphiz_dy, _ = dphiz[:, 0], dphiz[:, 1], dphiz[:, 2]

    # Compute velocities
    u = dphiz_dy - dphiy_dz  # shape (N,)
    v = dphix_dz - dphiz_dx
    w = dphiy_dx - dphix_dy

    # Stack into (N, 3)
    uvw_pred = torch.stack([u, v, w], dim=1)

    return uvw_pred

def boundary_mse_loss(uvw_boundary_pred, config):
    
    u_pred, v_pred, w_pred = uvw_boundary_pred[..., 0], uvw_boundary_pred[..., 1], uvw_boundary_pred[..., 2]

    mse_u = (u_pred) ** 2
    mse_v = (v_pred) ** 2
    mse_w = (w_pred) ** 2

    mse_bound = torch.mean(config.training.u_weight*mse_u + config.training.v_weight*mse_v + config.training.w_weight*mse_w)

    if config.training.pressure_in_boundary_loss:
        p_pred = uvw_boundary_pred[..., 3]
        mse_p = (p_pred) ** 2
        mse_bound_p = torch.mean(mse_p)
        mse_bound += config.training.p_weight*mse_bound_p

    return mse_bound

def compute_gradient(outputs, inputs, grad_dim):
    # Function to compute derivatives
    grad_outputs = torch.ones_like(outputs)
    return torch.autograd.grad(outputs=outputs, inputs=inputs,
                                grad_outputs=grad_outputs, # Specify dimension - compute gradient of each element
                                create_graph=True,  # Allows for higher order gradients
                                retain_graph=True,  # Allows to compute more gradients on the same graph
                                only_inputs=True)[0][..., grad_dim]

def navier_stokes_loss(uvw_pred, xyz_collocation, standardization_factors, config):

    # Unpack velocity and pressure
    u, v, w, p = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2], uvw_pred[..., 3]

    # First derivatives
    du_dt = compute_gradient(u, xyz_collocation, 0)
    du_dx = compute_gradient(u, xyz_collocation, 1)
    du_dy = compute_gradient(u, xyz_collocation, 2)
    du_dz = compute_gradient(u, xyz_collocation, 3)

    dv_dt = compute_gradient(v, xyz_collocation, 0)
    dv_dx = compute_gradient(v, xyz_collocation, 1)
    dv_dy = compute_gradient(v, xyz_collocation, 2)
    dv_dz = compute_gradient(v, xyz_collocation, 3)

    dw_dt = compute_gradient(w, xyz_collocation, 0)
    dw_dx = compute_gradient(w, xyz_collocation, 1)
    dw_dy = compute_gradient(w, xyz_collocation, 2)
    dw_dz = compute_gradient(w, xyz_collocation, 3)

    dp_dx = compute_gradient(p, xyz_collocation, 1)
    dp_dy = compute_gradient(p, xyz_collocation, 2)
    dp_dz = compute_gradient(p, xyz_collocation, 3)

    # Second derivatives
    d2u_dx2 = compute_gradient(du_dx, xyz_collocation, 1)
    d2u_dy2 = compute_gradient(du_dy, xyz_collocation, 2)
    d2u_dz2 = compute_gradient(du_dz, xyz_collocation, 3)

    d2v_dx2 = compute_gradient(dv_dx, xyz_collocation, 1)
    d2v_dy2 = compute_gradient(dv_dy, xyz_collocation, 2)
    d2v_dz2 = compute_gradient(dv_dz, xyz_collocation, 3)

    d2w_dx2 = compute_gradient(dw_dx, xyz_collocation, 1)
    d2w_dy2 = compute_gradient(dw_dy, xyz_collocation, 2)
    d2w_dz2 = compute_gradient(dw_dz, xyz_collocation, 3)
    
    # Extract constants
    U = config.constants.U
    L = config.constants.L
    rho = config.constants.rho
    mu = config.constants.mu
    Re = (rho*U*L)/mu # Reynolds number

    # density: 1.06, dynamic_viscosity: 0.035
    # rho: 1.06, mu: 0.035

    # Re = 

    if config.coords_normalization == "standardize":
        # Extract standardization factors
        _, std_t, _, std_x, _, std_y, _, std_z = standardization_factors
    else:
        std_t, std_x, std_y, std_z = 1.0, 1.0, 1.0, 1.0

    # Calculate residuals based on Navier-Stokes Equations
    momentum_u = (
        (1 / std_t) * du_dt 
        + (1 / std_x) * (u * du_dx) + (1 / std_y) * (v * du_dy) + (1 / std_z) * (w * du_dz) 
        + (1 / std_x) * dp_dx 
        - (1/Re) * ((1 / std_x**2) * d2u_dx2 + (1 / std_y**2) * d2u_dy2 + (1 / std_z**2) * d2u_dz2)
    )    

    momentum_v = (
        (1 / std_t) * dv_dt 
        + (1 / std_x) * (u * dv_dx) + (1 / std_y) * (v * dv_dy) + (1 / std_z) * (w * dv_dz) 
        + (1 / std_y) * dp_dy 
        - (1/Re) * ((1 / std_x**2) * d2v_dx2 + (1 / std_y**2) * d2v_dy2 + (1 / std_z**2) * d2v_dz2)
    ) 

    momentum_w = (
        (1 / std_t) * dw_dt 
        + (1 / std_x) * (u * dw_dx) + (1 / std_y) * (v * dw_dy) + (1 / std_z) * (w * dw_dz) 
        + (1 / std_z) * dp_dz 
        - (1/Re) * ((1 / std_x**2) * d2w_dx2 + (1 / std_y**2) * d2w_dy2 + (1 / std_z**2) * d2w_dz2)
    ) 

    # Calculate divergence
    div = (du_dx / std_x) + (dv_dy / std_y) + (dw_dz / std_z)

    # Calculate MSE
    momentum_loss_u = momentum_u ** 2
    momentum_loss_v = momentum_v ** 2
    momentum_loss_w = momentum_w ** 2

    momentum_loss = torch.mean(momentum_loss_u + momentum_loss_v + momentum_loss_w)
    div_loss = torch.mean(div ** 2)

    return momentum_loss, div_loss

def divergence_loss(uvw_pred, xyz_collocation, standardization_factors, config):

    # Unpack velocity and pressure
    u, v, w = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2]

    if config.setup.include_time:
        # [t, x, y, z].
        x_dim, y_dim, z_dim = 1, 2, 3
    else:
        # [x, y, z].
        x_dim, y_dim, z_dim = 0, 1, 2

    # First derivatives
    du_dx = compute_gradient(u, xyz_collocation, x_dim)
    dv_dy = compute_gradient(v, xyz_collocation, y_dim)
    dw_dz = compute_gradient(w, xyz_collocation, z_dim)

    if config.coords_normalization == "standardize":
        if config.setup.include_time:
            _, _, _, std_x, _, std_y, _, std_z = standardization_factors
        else:
            _, std_x, _, std_y, _, std_z = standardization_factors
    else:
        std_x, std_y, std_z = 1.0, 1.0, 1.0

    # Calculate divergence
    div = (du_dx / std_x) + (dv_dy / std_y) + (dw_dz / std_z)

    # Calculate loss
    div_loss = torch.mean(div ** 2)

    return div_loss


def five_point_encoding_cartesian(u, v, w, venc, m=None):
    """
    Five-point balanced encoding per Johnson & Markl (2010), Table 1.
    Encoding matrix rows correspond to the four non-flow-compensated points
    (tetrahedral balanced) plus the flow-compensated reference (M1=0).

    Phase: phi_r = pi/venc * (k_r . v), where k_r in {-1,0,+1}^3.

    Returns S_real, S_imag each of shape (N, 5).
    """
    # Encoding directions from Table 1 (Johnson & Markl 2010)
    # Points 0-3: balanced tetrahedral (non-flow-compensated)
    # Point 4:    flow-compensated reference (M1 = 0)
    k_vecs = torch.tensor([
        [-1.0, -1.0, -1.0],   # r=0
        [+1.0, +1.0, -1.0],   # r=1
        [+1.0, -1.0, +1.0],   # r=2
        [-1.0, +1.0, +1.0],   # r=3
        [ 0.0,  0.0,  0.0],   # r=4  flow-compensated reference
    ], dtype=u.dtype, device=u.device)  # (5, 3)

    # Phase: phi = (pi / venc) * (k . v)
    # This follows from gamma * Delta_M1 * v = pi when v = venc and |k|=1
    uvw = torch.stack([u, v, w], dim=-1)          # (N, 3)
    phi = (torch.pi / venc) * (uvw @ k_vecs.T)   # (N, 5)

    if m is None:
        mag = torch.ones(u.shape[0], 1, dtype=u.dtype, device=u.device)
    else:
        mag = m.unsqueeze(-1)                      # (N, 1)

    S_real = mag * torch.cos(phi)                  # (N, 5)
    S_imag = mag * torch.sin(phi)                  # (N, 5)  note: +sin, not -sin

    return S_real, S_imag



def fathi_five_point_loss(uvw_data, uvw_pred, config, m_mr=None, m_nn=None):
    """
    Fathi et al. (2020) data-fidelity loss using five-point balanced encoding.
 
    Both measured and predicted velocities are denormalized to physical units,
    converted to complex Cartesian images, and the MSE in Cartesian space is
    computed (Eq. 18, without Gaussian filter M, i.e. M = identity).
 
    Args:
        uvw_data : (N, 3+) measured velocities (normalised). First 3 cols = u,v,w.
        uvw_pred : (N, 3+) predicted velocities (normalised). First 3 cols = u,v,w.
        config   : training config.
        m_mr     : (N,) measured magnitude. If None, assumes m_MR = 1.
        m_nn     : (N,) predicted magnitude. If None, assumes m_NN = 1.
    """
    U_max = config.U_max
    venc  = config.constants.venc
 
    u_pred = uvw_pred[..., 0] * U_max
    v_pred = uvw_pred[..., 1] * U_max
    w_pred = uvw_pred[..., 2] * U_max
 
    u_data = uvw_data[..., 0] * U_max
    v_data = uvw_data[..., 1] * U_max
    w_data = uvw_data[..., 2] * U_max
 
    S_nn_real, S_nn_imag = five_point_encoding_cartesian(u_pred, v_pred, w_pred, venc, m=m_nn)
    S_mr_real, S_mr_imag = five_point_encoding_cartesian(u_data, v_data, w_data, venc, m=m_mr)
 
    diff_real = S_nn_real - S_mr_real   # (N, 5)
    diff_imag = S_nn_imag - S_mr_imag   # (N, 5)
 
    return torch.mean(diff_real ** 2 + diff_imag ** 2)


def data_loss_fn(model, xyz_data, uvw_data, mask, config):
    
    # Predict data points
    uvw_pred = model(xyz_data)

    if config.training.use_vector_potential:
        # Transform potential to velocities
        uvw_pred = vector_potential_fn(uvw_pred, xyz_data)

    # Calculate loss
    if config.training.use_mse:
        if config.setup.fluid_region:
            return mse_loss(uvw_pred, uvw_data, config)
        else:
            return fluid_weighted_mse_loss(uvw_pred, uvw_data, mask, config)    
    else:
        raise ValueError("No data loss specified, check config.training")

def physics_loss_fn(model, xyz_collocation, standardization_factors, config):

    # Predict collocation points
    uvw_pred = model(xyz_collocation)

    # Calculate loss
    if config.training.use_navier_stokes:
        momentum_loss, div_loss = navier_stokes_loss(uvw_pred, xyz_collocation, standardization_factors, config)
        physics_loss = momentum_loss + div_loss
    elif config.training.use_divergence:
        div_loss = divergence_loss(uvw_pred, xyz_collocation, standardization_factors, config)
        physics_loss = div_loss
    else:
        raise ValueError("No physics loss specified, check config.training")

    return config.training.physics_weight*physics_loss

def boundary_loss_fn(model, xyz_boundary, config):
    if config.training.use_vector_potential:
        xyz_boundary.requires_grad_(True)

    # Predict boundary points
    uvw_pred_boundary = model(xyz_boundary)

    if config.training.use_vector_potential:
        uvw_pred_boundary = vector_potential_fn(uvw_pred_boundary, xyz_boundary)
    
    # Calculate loss
    if config.training.use_boundary_mse:
        return boundary_mse_loss(uvw_pred_boundary, config)
    else:
        raise ValueError("No boundary loss specified, check config.training")

def compute_grad_norms(model, loss_fn, inputs, *args):
    
    # Reset gradients
    model.zero_grad()  

    # Compute loss
    loss = loss_fn(model, inputs, *args)
    loss.backward() 

    # Initialize list for gradients
    all_grads = []

    # Loop and collect gradients
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))

    # Concatenate gradients
    all_grads = torch.cat(all_grads)

    # Compute L2 norm
    grad_norm = all_grads.norm(p=2).item()

    return grad_norm

def update_loss_weights(config, model, loss_weights, it, xyz_data_batch, uvw_data_batch, mask, xyz_collocation_batch, xyz_boundary_batch, standardization_factors):
    
    # Initalize if loss_weights is None
    if loss_weights is None: 
        if config.sample_boundary:
            loss_weights = np.array([1.0, 1.0, 1.0])
        else:
            loss_weights = np.array([1.0, 1.0])
    
    # Return as is if data pre-training
    if it < config.training.epochs_before_PDE:
        return loss_weights
    
    # Compute gradient norms
    grad_norm_data = compute_grad_norms(model, data_loss_fn, xyz_data_batch, uvw_data_batch, mask, config)
    grad_norm_physics = compute_grad_norms(model, physics_loss_fn, xyz_collocation_batch, standardization_factors, config)
    if config.sample_boundary:
        grad_norm_boundary = compute_grad_norms(model, boundary_loss_fn, xyz_boundary_batch, config)

    # Collect gradient norms
    if config.sample_boundary:
        grad_norms = [grad_norm_data, grad_norm_physics, grad_norm_boundary]
    else:
        grad_norms = [grad_norm_data, grad_norm_physics]

    # Calculate and update weights
    weights_old = loss_weights
    weights_new = np.array([sum(grad_norms) / (grad_norm + 1e-7) for grad_norm in grad_norms])
    loss_weights = config.training.alpha*weights_old + (1-config.training.alpha)*weights_new

    return loss_weights

def compute_data_loss(config, model, xyz_data, uvw_data, mask):
    

    # Predict data points
    uvw_pred = model(xyz_data)
    # Transform potential to velocity
    if config.training.use_vector_potential:
        uvw_pred = vector_potential_fn(uvw_pred, xyz_data)
    
    use_five = getattr(config.training, "use_fathi_five_point_loss", False)
    
    # Compute data loss
    if use_five:
        data_loss = fathi_five_point_loss(uvw_data, uvw_pred, config,m_mr=None,m_nn=None)
        # data_loss = four_point_loss(uvw_data, uvw_pred, config,m_mr=None,m_nn=None)

    elif config.training.use_mse:
        if config.setup.fluid_region:
            data_loss = mse_loss(uvw_pred, uvw_data, config)
        else:
            data_loss = fluid_weighted_mse_loss(uvw_pred, uvw_data, mask, config)
    
    elif config.training.use_cosine:
        if config.setup.fluid_region:
            data_loss = cosine_loss(uvw_pred, uvw_data, config)
        else:
            raise ValueError("No fluid_weighted_cosine_loss implemented yet.")
            #TODO
            # data_loss = fluid_weighted_cosine_loss(uvw_pred, uvw_data, mask, config)
    else:
        raise ValueError("No recognized data loss mode. Check config.training.use_mse or other flags.")

    return data_loss, uvw_pred

def compute_physics_loss(config, iter, model, xyz_collocation, xyz_data, standardization_factors):

    # Initialize dictionary for possible losses
    losses = {
        "physics_loss": torch.tensor(0.0),
        "momentum_loss": torch.tensor(0.0),
        "div_loss": torch.tensor(0.0),
        "physics_loss_data": torch.tensor(0.0),
        "momentum_loss_data": torch.tensor(0.0),
        "div_loss_data": torch.tensor(0.0),
    }

    # Skip if no physics loss
    if not config.training.use_physics_loss:
        return losses  # all remain None

    # Skip if PDE not active yet
    if iter < config.training.epochs_before_PDE:
        zero_t = torch.tensor(0.0)
        losses["physics_loss"] = zero_t
        losses["momentum_loss"] = zero_t
        losses["div_loss"] = zero_t
        losses["physics_loss_data"] = zero_t
        losses["momentum_loss_data"] = zero_t
        losses["div_loss_data"] = zero_t
        return losses

    # Predict collocation points
    uvw_pred_physics = model(xyz_collocation)
    if config.training.physics_loss_on_data_points:
        uvw_pred_data = model(xyz_data)

    # Navier-Stokes
    if config.training.use_navier_stokes:
        momentum_loss, div_loss = navier_stokes_loss(uvw_pred_physics, xyz_collocation, standardization_factors, config)
        physics_loss = momentum_loss + div_loss

        if config.training.physics_loss_on_data_points:
            momentum_loss_data, div_loss_data = navier_stokes_loss(uvw_pred_data, xyz_data, standardization_factors, config)
            physics_loss_data = momentum_loss_data + div_loss_data

        losses["physics_loss"] = physics_loss
        losses["momentum_loss"] = momentum_loss
        losses["div_loss"] = div_loss

        if config.training.physics_loss_on_data_points:
            losses["physics_loss_data"] = physics_loss_data
            losses["momentum_loss_data"] = momentum_loss_data
            losses["div_loss_data"] = div_loss_data

        return losses

    # Divergence only
    elif config.training.use_divergence:
        div_loss = divergence_loss(uvw_pred_physics, xyz_collocation, standardization_factors, config)
        if config.training.physics_loss_on_data_points:
            div_loss_data = divergence_loss(uvw_pred_data, xyz_data, standardization_factors, config)

        losses["physics_loss"] = div_loss
        losses["div_loss"] = div_loss

        if config.training.physics_loss_on_data_points:
            losses["physics_loss_data"] = div_loss_data
            losses["div_loss_data"] = div_loss_data
            
        return losses

    else:
        # Implement alternative loss here
        raise ValueError("No physics loss specified, check config.training")

def compute_boundary_loss(config, model, xyz_boundary):

    bound_loss = torch.tensor(0.0)

    # Skip if no boundary loss
    if not config.sample_boundary:
        return bound_loss  # all remain None
    
    if config.training.use_vector_potential:
        xyz_boundary.requires_grad_(True)

    # Predict boundary points
    uvw_pred = model(xyz_boundary)

    if config.training.use_vector_potential:
        uvw_pred = vector_potential_fn(uvw_pred, xyz_boundary)
    
    # Calculate loss
    if config.training.use_boundary_mse:
        return boundary_mse_loss(uvw_pred, config)
    else:
        # Implement alternative loss here
        raise ValueError("No boundary loss specified, check config.training")
