from __future__ import annotations

import numpy as np
import pytest
import torch

import planner
from model_streamvln import SigLipImageProcessor


@pytest.fixture(scope="module")
def processor() -> SigLipImageProcessor:
    return SigLipImageProcessor()


def test_frame_becomes_bf16_siglip_pixels(processor: SigLipImageProcessor, frames: list[np.ndarray]):
    pixels = planner.preprocess(processor, frames[0])
    assert pixels.shape == (3, 384, 384)
    assert pixels.dtype == torch.bfloat16
    assert -1.0 <= pixels.min().item() < pixels.max().item() <= 1.0


def test_frame_is_squashed_not_cropped(processor: SigLipImageProcessor):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :64] = (255, 0, 0)
    pixels = planner.preprocess(processor, rgb).float()
    assert pixels[0, :, :20].mean().item() == pytest.approx(1.0, abs=0.02)
    assert pixels[0, :, 60:].mean().item() == pytest.approx(-1.0, abs=0.02)


def test_alpha_channel_is_dropped(processor: SigLipImageProcessor, frames: list[np.ndarray]):
    rgba = np.concatenate([frames[1], np.full((480, 640, 1), 255, dtype=np.uint8)], axis=-1)
    assert torch.equal(planner.preprocess(processor, rgba), planner.preprocess(processor, frames[1]))


def test_processor_defaults_are_siglip_so400m_384(processor: SigLipImageProcessor):
    assert processor.size == (384, 384)
    assert processor.image_mean == processor.image_std == (0.5, 0.5, 0.5)
