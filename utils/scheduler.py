import torch

def linear_beta_schedule(timesteps):
    """Create a linear schedule for beta."""
    return torch.linspace(1e-4, 0.02, timesteps)
