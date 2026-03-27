"""Shared utilities for neural network initialization and common patterns."""

import torch.nn as nn
from einops import rearrange


def init_module_weights(m, std: float = 0.02):
    """
    Initialize weights for common layer types using truncated normal distribution.

    This is a unified weight initialization function used across the codebase.
    Apply it via module.apply(init_module_weights) or as a method wrapper.

    Args:
        m: PyTorch module to initialize
        std: Standard deviation for truncated normal initialization (default: 0.02)
    """
    if isinstance(
        m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    ):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TemporalBatchMixin:
    """
    Mixin class that handles automatic temporal batching for 3D/4D/5D tensors.

    This mixin provides a unified forward() method that:
    - For 5D tensors [B, C, T, H, W]: flattens temporal dim, applies _forward(), restores shape
      (output may be spatial [B, C_out, T, H_out, W_out] or flat [B, C_out, T])
    - For 3D tensors [B, C, T]: flattens temporal dim, applies _forward(), restores shape
    - For 4D tensors [B, C, H, W]: directly applies _forward()

    Subclasses must implement _forward(self, x) for the base (non-temporal) tensor shape.
    """

    def _forward(self, x):
        """
        Process a single-frame tensor. Must be implemented by subclasses.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, C]

        Returns:
            Output tensor of shape [B, C_out, H_out, W_out] or [B, C_out]
        """
        raise NotImplementedError("Subclasses must implement _forward()")

    def forward(self, x):
        """
        Forward pass supporting 3D, 4D, and 5D tensors.

        Args:
            x: Input tensor of shape [B, C, H, W], [B, C, T, H, W], or [B, C, T]

        Returns:
            Output tensor with same batch and temporal dimensions as input.
            For 5D input: [B, C_out, T, H_out, W_out] or [B, C_out, T] depending on _forward output.
            For 3D input: [B, C_out, T].
            For 4D input: [B, C_out, H_out, W_out].
        """
        assert x.ndim in [
            3,
            4,
            5,
        ], "Supports only 3D [B, C, T], 4D [B, C, H, W], or 5D [B, C, T, H, W] tensors"
        if x.ndim == 5:
            b = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            if len(out.shape) == 2:
                out = rearrange(out, "(b t) c -> b c t", b=b)
            else:    
                out = rearrange(out, "(b t) c h w -> b c t h w", b=b)
            return out
        if x.ndim == 3:
            b = x.shape[0]
            x = rearrange(x, "b c t -> (b t) c")
            out = self._forward(x)
            out = rearrange(out, "(b t) c -> b c t", b=b)
            return out 
        else:
            return self._forward(x)
