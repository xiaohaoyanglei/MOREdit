"""Simple LoRA injection utilities for Flux cross-attention modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from diffusers.models.transformers.transformer_flux import FluxAttention


class LoRALinear(torch.nn.Module):
    """Wrap a Linear-like module with a low-rank adapter."""

    def __init__(self, base: torch.nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        original_bias = base.bias
        self.base = base
        self.base.weight.requires_grad_(False)
        if original_bias is not None:
            self.base.bias = None

        self.rank = rank
        self.scale = alpha / max(rank, 1)

        in_features = base.in_features
        out_features = base.out_features

        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.bias = torch.nn.Parameter(original_bias.detach().clone()) if original_bias is not None else None

        target_device = base.weight.device
        target_dtype = base.weight.dtype
        self.lora_down.to(device=target_device, dtype=target_dtype)
        self.lora_up.to(device=target_device, dtype=target_dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(device=target_device, dtype=target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        result = result + self.lora_up(self.lora_down(x)) * self.scale
        if self.bias is not None:
            result = result + self.bias
        return result

    @property
    def lora_parameters(self) -> Iterable[torch.nn.Parameter]:
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()


@dataclass
class LoRAInjectionConfig:
    rank: int = 16
    alpha: float = 32.0
    attn_block_prefixes: Tuple[str, ...] = (
        "transformer_blocks.14",
        "transformer_blocks.15",
        "transformer_blocks.16",
    )
    target_projections: Tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    )


def inject_lora_into_attention(
    transformer: torch.nn.Module,
    config: LoRAInjectionConfig,
) -> List[LoRALinear]:
    """Inject LoRA modules into FluxAttention projections."""

    lora_modules: List[LoRALinear] = []

    for module_path, module in transformer.named_modules():
        if not isinstance(module, FluxAttention):
            continue
        if not any(module_path.startswith(prefix) for prefix in config.attn_block_prefixes):
            continue

        for target in config.target_projections:
            if target.startswith("to_out"):
                target_module = module.to_out[0]
                wrapper = LoRALinear(target_module, config.rank, config.alpha)
                wrapper.lora_name = f"{module_path}.{target}"
                module.to_out[0] = wrapper
            else:
                target_module = getattr(module, target, None)
                if target_module is None or not isinstance(target_module, torch.nn.Linear):
                    continue
                wrapper = LoRALinear(target_module, config.rank, config.alpha)
                wrapper.lora_name = f"{module_path}.{target}"
                setattr(module, target, wrapper)

            lora_modules.append(wrapper)

    if not lora_modules:
        raise RuntimeError("No LoRA modules were injected. Check attn prefixes and target projections.")

    return lora_modules


