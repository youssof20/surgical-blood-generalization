"""U-Net with pretrained ResNet34 encoder (segmentation-models-pytorch)."""

from __future__ import annotations

import segmentation_models_pytorch as smp


def build_unet() -> smp.Unet:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
