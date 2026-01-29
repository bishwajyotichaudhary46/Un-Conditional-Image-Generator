import torch
def add_noise(
                
        original_samples: torch.FloatTensor,
        timestep: torch.FloatTensor,
        noise: torch.FloatTensor
    ) -> torch.FloatTensor:
        x_t = (1 - timestep) * original_samples + timestep * noise
        return x_t