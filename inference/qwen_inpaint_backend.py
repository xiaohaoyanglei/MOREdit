"""Qwen inpaint/edit backend wrapper for MOREdit soft-mask pipeline."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image, ImageOps

from ..trainer import _to_dtype


class QwenInpaintBackend:
    """Thin wrapper around Qwen diffusers pipelines for mask-based editing."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype | str] = None,
        mask_invert: Optional[bool] = None,
        force_binary_mask: bool = True,
        mode: str = "inpaint",
        controlnet_path: Optional[str] = None,
    ) -> None:
        default_path = os.getenv("QWEN_EDIT_MODEL_PATH", "/workspace/models_weight/Qwen-Image-Edit-2511")
        self.model_path = Path(model_path or default_path)

        # ControlNet support: default path from env, enable by default for inpaint mode
        default_controlnet = os.getenv("QWEN_CONTROLNET_PATH", "/workspace/models_weight/Qwen-Image-ControlNet-Inpainting")
        self.controlnet_path = Path(controlnet_path or default_controlnet) if controlnet_path != "" else None
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Qwen model path not found: {self.model_path} (set QWEN_EDIT_MODEL_PATH if needed)"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None:
            self.dtype = torch.bfloat16
        else:
            self.dtype = _to_dtype(dtype) if isinstance(dtype, str) else dtype
        if mask_invert is None:
            env = os.getenv("QWEN_MASK_INVERT", "0").strip().lower()
            mask_invert = env in ("1", "true", "yes", "y", "on")
        self.mask_invert = bool(mask_invert)
        self.force_binary_mask = bool(force_binary_mask)
        self.cpu_offload = os.getenv("QWEN_CPU_OFFLOAD", "0").strip().lower() in ("1", "true", "yes", "y", "on")
        self.device_map = os.getenv("QWEN_DEVICE_MAP", "").strip()
        self.mode = str(mode).strip().lower()

        self._pipeline = None
        self._pipeline_name = None
        self._controlnet = None
        self._using_controlnet = False
        self.last_mask_used: Optional[Image.Image] = None
        self.last_image_resized: Optional[Image.Image] = None

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import diffusers  # type: ignore
        except Exception as exc:
            raise ImportError(f"diffusers is required for Qwen backend: {exc}") from exc

        # Try ControlNet pipeline first for inpaint mode
        if self.mode == "inpaint" and self.controlnet_path is not None and self.controlnet_path.exists():
            print(f"[qwen] attempting to load ControlNet from {self.controlnet_path}")
            try:
                controlnet_cls = getattr(diffusers, "QwenImageControlNetModel", None)
                pipeline_cls = getattr(diffusers, "QwenImageControlNetInpaintPipeline", None)

                if controlnet_cls is not None and pipeline_cls is not None:
                    # Load ControlNet model
                    self._controlnet = controlnet_cls.from_pretrained(
                        str(self.controlnet_path),
                        torch_dtype=self.dtype,
                    )
                    print(f"[qwen] loaded ControlNet model from {self.controlnet_path}")

                    # Load pipeline with ControlNet
                    if self.device_map:
                        pipe = pipeline_cls.from_pretrained(
                            str(self.model_path),
                            controlnet=self._controlnet,
                            torch_dtype=self.dtype,
                            device_map=self.device_map,
                        )
                    else:
                        pipe = pipeline_cls.from_pretrained(
                            str(self.model_path),
                            controlnet=self._controlnet,
                            torch_dtype=self.dtype,
                        )

                    self._pipeline = pipe
                    self._pipeline_name = "QwenImageControlNetInpaintPipeline"
                    self._using_controlnet = True
                    print(f"[qwen] ✓ using ControlNet inpaint pipeline")
                else:
                    print(f"[qwen] ControlNet classes not found in diffusers, falling back to base pipeline")
            except Exception as exc:
                print(f"[qwen] failed to load ControlNet (will fall back to base pipeline): {exc}")
                self._controlnet = None
                self._using_controlnet = False

        # Fall back to base pipeline if ControlNet not loaded
        if self._pipeline is None:
            if self.mode == "edit":
                candidates = [
                    "QwenImageEditPlusPipeline",
                    "QwenImageEditPipeline",
                ]
            else:
                candidates = [
                    "QwenImageInpaintPipeline",
                ]
            errors = []
            for name in candidates:
                cls = getattr(diffusers, name, None)
                if cls is None:
                    continue
                try:
                    if self.device_map:
                        pipe = cls.from_pretrained(
                            str(self.model_path),
                            torch_dtype=self.dtype,
                            device_map=self.device_map,
                        )
                    else:
                        pipe = cls.from_pretrained(str(self.model_path), torch_dtype=self.dtype)
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                    continue
                self._pipeline = pipe
                self._pipeline_name = name
                break

            if self._pipeline is None:
                detail = "; ".join(errors) if errors else "no candidate class found"
                raise RuntimeError(f"Failed to load Qwen pipeline from {self.model_path}: {detail}")

        if self.device_map:
            print(f"[qwen] device_map={self.device_map}")
        elif self.cpu_offload and hasattr(self._pipeline, "enable_model_cpu_offload"):
            self._pipeline.enable_model_cpu_offload()
            print("[qwen] enable_model_cpu_offload=on")
        else:
            self._pipeline.to(self.device)
        if hasattr(self._pipeline, "set_progress_bar_config"):
            self._pipeline.set_progress_bar_config(disable=True)

        pipeline_type = "ControlNet " if self._using_controlnet else ""
        print(f"[qwen] using diffusers {pipeline_type}{self._pipeline_name} from {self.model_path}")

    @staticmethod
    def _normalize_out_size(out_size: Optional[Tuple[int, int] | int], fallback: Tuple[int, int]) -> Tuple[int, int]:
        if out_size is None:
            return fallback
        if isinstance(out_size, int):
            return (int(out_size), int(out_size))
        return (int(out_size[0]), int(out_size[1]))

    def run_inpaint(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        prompt: str,
        seed: Optional[int],
        steps: int,
        guidance: float,
        out_size: Optional[Tuple[int, int] | int],
        negative_prompt: Optional[str] = None,
        true_cfg_scale: Optional[float] = None,
        strength: Optional[float] = None,
        controlnet_conditioning_scale: Optional[float] = None,
    ) -> Image.Image:
        if self.mode != "inpaint":
            raise RuntimeError(f"Qwen backend mode is {self.mode}, cannot run inpaint.")
        self._load_pipeline()

        image = image_pil.convert("RGB")
        mask = mask_pil.convert("L")
        width, height = self._normalize_out_size(out_size, image.size)

        if image.size != (width, height):
            image = image.resize((width, height), Image.BICUBIC)
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.NEAREST)

        if self.force_binary_mask:
            mask = mask.point(lambda p: 255 if p >= 128 else 0)
        if self.mask_invert:
            mask = ImageOps.invert(mask)

        self.last_image_resized = image.copy()
        self.last_mask_used = mask.copy()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        params = inspect.signature(self._pipeline.__call__).parameters
        kwargs = {}

        if self._using_controlnet:
            # Qwen ControlNet inpaint expects `control_image`/`control_mask`.
            if "control_image" in params:
                kwargs["control_image"] = image
            elif "image" in params:
                kwargs["image"] = image
            elif "images" in params:
                kwargs["images"] = image
            else:
                raise RuntimeError(
                    "[qwen] ControlNet pipeline __call__ does not accept control_image/image/images."
                )
        else:
            if "image" in params:
                kwargs["image"] = image
            elif "images" in params:
                kwargs["images"] = image
            else:
                raise RuntimeError("[qwen] pipeline __call__ does not accept an image parameter.")

        # ControlNet pipelines use 'control_mask' instead of 'mask'
        mask_param = None
        if self._using_controlnet:
            # ControlNet-specific parameters
            for candidate in ("control_mask", "mask", "mask_image"):
                if candidate in params:
                    mask_param = candidate
                    break
        else:
            # Base pipeline parameters
            for candidate in ("mask", "mask_image", "mask_image_l"):
                if candidate in params:
                    mask_param = candidate
                    break

        if mask_param is None:
            raise RuntimeError("[qwen] pipeline __call__ does not accept a mask parameter; cannot inpaint.")
        kwargs[mask_param] = mask

        if "prompt" in params:
            kwargs["prompt"] = prompt
        if negative_prompt and "negative_prompt" in params:
            kwargs["negative_prompt"] = negative_prompt
        if true_cfg_scale is not None and "true_cfg_scale" in params:
            kwargs["true_cfg_scale"] = float(true_cfg_scale)
        if strength is not None and "strength" in params:
            kwargs["strength"] = float(strength)
        if "generator" in params and generator is not None:
            kwargs["generator"] = generator
        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = int(steps)
        elif "num_steps" in params:
            kwargs["num_steps"] = int(steps)
        elif "steps" in params:
            kwargs["steps"] = int(steps)

        if "guidance_scale" in params:
            kwargs["guidance_scale"] = float(guidance)
        elif "guidance" in params:
            kwargs["guidance"] = float(guidance)
        elif "cfg_scale" in params:
            kwargs["cfg_scale"] = float(guidance)

        if "height" in params:
            kwargs["height"] = int(height)
        if "width" in params:
            kwargs["width"] = int(width)

        if controlnet_conditioning_scale is not None and "controlnet_conditioning_scale" in params:
            kwargs["controlnet_conditioning_scale"] = float(controlnet_conditioning_scale)

        controlnet_info = " [ControlNet]" if self._using_controlnet else ""
        print(
            f"[qwen] run{controlnet_info}: size={width}x{height}, steps={steps}, guidance={guidance}, "
            f"mask_param={mask_param}, mask_invert={self.mask_invert}"
        )

        with torch.inference_mode():
            output = self._pipeline(**kwargs)

        if hasattr(output, "images"):
            return output.images[0]
        if isinstance(output, (list, tuple)) and output:
            return output[0]
        if isinstance(output, Image.Image):
            return output
        raise RuntimeError(f"[qwen] unexpected output type: {type(output)}")

    def run_edit(
        self,
        image_pil: Image.Image,
        prompt: str,
        seed: Optional[int],
        steps: int,
        guidance: float,
        out_size: Optional[Tuple[int, int] | int],
        negative_prompt: Optional[str] = None,
        true_cfg_scale: Optional[float] = None,
    ) -> Image.Image:
        if self.mode != "edit":
            raise RuntimeError(f"Qwen backend mode is {self.mode}, cannot run edit.")
        self._load_pipeline()

        image = image_pil.convert("RGB")
        width, height = self._normalize_out_size(out_size, image.size)
        if image.size != (width, height):
            image = image.resize((width, height), Image.BICUBIC)
        self.last_image_resized = image.copy()
        self.last_mask_used = None

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        params = inspect.signature(self._pipeline.__call__).parameters
        kwargs = {}
        if "image" in params:
            kwargs["image"] = image
        elif "images" in params:
            kwargs["images"] = image
        else:
            raise RuntimeError("[qwen] pipeline __call__ does not accept an image parameter.")

        if "prompt" in params:
            kwargs["prompt"] = prompt
        if negative_prompt and "negative_prompt" in params:
            kwargs["negative_prompt"] = negative_prompt
        if true_cfg_scale is not None and "true_cfg_scale" in params:
            kwargs["true_cfg_scale"] = float(true_cfg_scale)
        if "generator" in params and generator is not None:
            kwargs["generator"] = generator
        if "num_inference_steps" in params:
            kwargs["num_inference_steps"] = int(steps)
        elif "num_steps" in params:
            kwargs["num_steps"] = int(steps)
        elif "steps" in params:
            kwargs["steps"] = int(steps)

        if "guidance_scale" in params:
            kwargs["guidance_scale"] = float(guidance)
        elif "guidance" in params:
            kwargs["guidance"] = float(guidance)
        elif "cfg_scale" in params:
            kwargs["cfg_scale"] = float(guidance)

        if "height" in params:
            kwargs["height"] = int(height)
        if "width" in params:
            kwargs["width"] = int(width)

        print(
            f"[qwen] run(edit): size={width}x{height}, steps={steps}, guidance={guidance}"
        )

        with torch.inference_mode():
            output = self._pipeline(**kwargs)

        if hasattr(output, "images"):
            return output.images[0]
        if isinstance(output, (list, tuple)) and output:
            return output[0]
        if isinstance(output, Image.Image):
            return output
        raise RuntimeError(f"[qwen] unexpected output type: {type(output)}")

    def run(self, *args, **kwargs) -> Image.Image:
        return self.run_inpaint(*args, **kwargs)
