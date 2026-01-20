"""Custom FlowMatch scheduler adapted for pointer training."""

import torch
import numpy as np
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """Matches the scheduler settings used by Flux Kontext training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

        with torch.no_grad():
            num_timesteps = 1000
            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)
            y_shifted = y - y.min()
            bsmntw = y_shifted * (num_timesteps / y_shifted.sum())
            hbsmntw = bsmntw.clone()
            hbsmntw[num_timesteps // 2 :] = hbsmntw[num_timesteps // 2 :].max()

            timesteps = torch.linspace(1000, 1, num_timesteps, device="cpu")

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw
            self.linear_timesteps_weights2 = hbsmntw

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2: bool = False, timestep_type: str = "linear") -> torch.Tensor:
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]
        if timestep_type == "weighted":
            weights = self.linear_timesteps_weights[step_indices]
        elif v2:
            weights = self.linear_timesteps_weights2[step_indices]
        else:
            weights = self.linear_timesteps_weights[step_indices]
        return weights

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        return (1.0 - t_01) * original_samples + t_01 * noise

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:  # noqa: D401
        return sample

    def set_train_timesteps(
        self,
        num_timesteps: int,
        device: torch.device,
        timestep_type: str = "linear",
        latents: torch.Tensor | None = None,
        patch_size: int = 1,
    ):
        self.timestep_type = timestep_type
        if timestep_type in {"linear", "weighted"}:
            timesteps = torch.linspace(1000, 1, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        elif timestep_type == "sigmoid":
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))
            timesteps = ((1 - t) * 1000)
            timesteps, _ = torch.sort(timesteps, descending=True)
            self.timesteps = timesteps.to(device=device)
            return timesteps
        elif timestep_type in {"flux_shift", "lumina2_shift", "shift"}:
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max),
                self._sigma_to_t(self.sigma_min),
                num_timesteps,
            )
            sigmas = timesteps / self.config.num_train_timesteps

            if self.config.use_dynamic_shifting:
                if latents is None:
                    raise ValueError("Latents must be provided when dynamic shifting is enabled.")
                h = latents.shape[2]
                w = latents.shape[3]
                image_seq_len = h * w // (patch_size**2)
                mu = calculate_shift(
                    image_seq_len,
                    self.config.get("base_image_seq_len", 256),
                    self.config.get("max_image_seq_len", 4096),
                    self.config.get("base_shift", 0.5),
                    self.config.get("max_shift", 1.16),
                )
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:
                sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)

            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps

            self.timesteps = timesteps.to(device=device)
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
            return timesteps
        elif timestep_type == "lognorm_blend":
            alpha = 0.75
            lognormal = LogNormal(loc=0, scale=0.333)
            t1 = lognormal.sample((int(num_timesteps * alpha),)).to(device)
            t1 = ((1 - t1 / t1.max()) * 1000)
            t2 = torch.linspace(1000, 1, int(num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))
            timesteps, _ = torch.sort(timesteps, descending=True)
            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device=device)
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return float(mu)


