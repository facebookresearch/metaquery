# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, List, Tuple
from collections.abc import Sequence

import torch
from PIL import Image
from diffusers.pipelines.pipeline_utils import BaseOutput
import numpy as np
from dataclasses import dataclass
from models.metaquery import MetaQuery


@dataclass
class MetaQueryPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray]
    # text: Optional[List[str]] = [""]


class MetaQueryPipeline(MetaQuery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_image(self, img: Union[str, Image.Image, torch.Tensor, np.ndarray],) -> Image.Image:
        """
        Normalise a image into a fresh ``PIL.Image`` (RGB / RGBA / L).

        Accepts
        --------
        - str             : local / remote path
        - PIL.Image       : any mode, auto-converted to RGB
        - torch.Tensor    : (H, W) or (C, H, W) with 1 / 3 / 4 channels
        - np.ndarray      : same shapes as tensor input

        Any other shape or dtype raises ``TypeError``.
        """
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 2:
                return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
            if arr.ndim == 3:
                h, w, c = arr.shape
                if c == 1:
                    arr = arr.squeeze(-1)
                    return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
                if c in (3, 4):
                    return Image.fromarray(arr.astype(np.uint8))
            raise TypeError(f"Unsupported numpy shape: {arr.shape}")
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.dim() == 2:
                if t.is_floating_point():
                    t = (t.clamp(0, 1) * 255)
                return Image.fromarray(t.to(torch.uint8).numpy(), mode="L").convert("RGB")
            if t.dim() == 3:
                if t.is_floating_point():
                    t = (t.clamp(0, 1) * 255)
                t = t.to(torch.uint8)
                c = t.shape[0]
                if c == 1:
                    arr = t.squeeze(0).numpy()
                    return Image.fromarray(arr, mode="L").convert("RGB")
                if c in (3, 4):
                    arr = t.permute(1, 2, 0).numpy()
                    mode = "RGB" if c == 3 else "RGBA"
                    return Image.fromarray(arr, mode=mode)
            raise TypeError(f"Unsupported tensor shape: {t.shape}")
        raise TypeError(f"Unsupported image type: {type(img)}")

    def _prepare_images(self, images):
        """Recursively convert nested image structures to lists of ``PIL.Image`` leaves."""
        def convert(elem):
            if elem is None:
                return None
            if isinstance(elem, (str, Image.Image, torch.Tensor, np.ndarray)):
                return self._load_image(elem)
            if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes, bytearray)):
                return [convert(e) for e in elem]
            raise TypeError(f"Unsupported image input type: {type(elem)}")
        return None if images is None else convert(images)

    @torch.no_grad()
    def __call__(
        self,
        caption: Optional[str] = "",
        image: Optional[
            Union[
                Image.Image,
                str,
                List[Union[Image.Image, str, torch.Tensor, List[Union[Image.Image, str, torch.Tensor]]]],
                torch.Tensor,
                List[torch.Tensor],
            ]
        ] = None,
        negative_prompt: Optional[str] = "",
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[MetaQueryPipelineOutput, Tuple]:

        images = self._prepare_images(image)

        samples = self.sample_images(
            caption=caption,
            input_images=images,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )
        if not return_dict:
            return (samples,)

        return MetaQueryPipelineOutput(images=samples)
