import math
import torch


def _complex_mul(x, y):
    xr, xi = x[..., 0], x[..., 1]
    yr, yi = y[..., 0], y[..., 1]
    return torch.stack([xr * yr - xi * yi, xr * yi + xi * yr], dim=-1)


def _complex_exp(theta):
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def freq_compensation(segment, freq, PROCESS_SAMPLING_RATE):
    # segment: (N, T, 2), freq: (N,)
    n, segment_length = segment.shape[0], segment.shape[1]
    index = torch.arange(segment_length, device=segment.device, dtype=segment.dtype).view(1, -1)
    phase = freq.view(n, 1) * (2.0 * math.pi / PROCESS_SAMPLING_RATE) * index
    rotator = _complex_exp(phase).view(n, segment_length, 2)
    return _complex_mul(segment, rotator)


def phase_compensation(segment, phase):
    # segment: (N, T, 2), phase: (N,)
    n, t, _ = segment.shape
    rotator = _complex_exp(-phase.view(n, 1)).view(n, 1, 2).repeat(1, t, 1)
    return _complex_mul(segment, rotator)
