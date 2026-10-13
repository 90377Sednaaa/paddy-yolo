"""Dynamic registration of custom Paddy-YOLO modules into Ultralytics.

This module injects custom Paddy-YOLO components (RGG, RGGBottleneck, RGGConv,
RepConv, GhostConv) into Ultralytics namespaces (`ultralytics.nn.modules`,
`ultralytics.nn.modules.block`, and `ultralytics.nn.tasks`) and updates `parse_model`
so that `RGG` is parsed using the C2f argument signature `[c1, c2, *args[1:]]`.

This enables loading Paddy-YOLO YAML architectures directly via
`YOLO("configs/paddy-yolo.yaml")` without modifying files in site-packages.
"""

import ast
import inspect
from copy import deepcopy
from typing import Dict, Type

import torch
import torch.nn as nn

import ultralytics.nn as ultralytics_nn
import ultralytics.nn.modules as ultralytics_modules
import ultralytics.nn.modules.block as ultralytics_block
import ultralytics.nn.tasks as ultralytics_tasks

from paddy_yolo.modules.dysample import DySample
from paddy_yolo.modules.ghostconv import GhostConv
from paddy_yolo.modules.repconv import RepConv
from paddy_yolo.modules.rgg import RGG, RGGBottleneck, RGGConv

# Registry of custom Paddy-YOLO modules to inject
CUSTOM_MODULES: Dict[str, Type[nn.Module]] = {
    "RGG": RGG,
    "RGGBottleneck": RGGBottleneck,
    "RGGConv": RGGConv,
    "RepConv": RepConv,
    "GhostConv": GhostConv,
    "DySample": DySample,
}

_PADDY_MODULES_REGISTERED = False


def _build_fallback_parse_model():
    """Build a standalone fallback parse_model function that includes RGG."""

    def fallback_parse_model(d, ch, verbose=True):
        """Parse a YOLO model.yaml dictionary into a PyTorch model including RGG."""
        legacy = True
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (
            d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
        )
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                ultralytics_tasks.LOGGER.warning(
                    f"WARNING \u26a0\ufe0f no model scale passed. Assuming scale='{scale}'."
                )
            depth, width, max_channels = scales[scale]

        if act:
            ultralytics_tasks.Conv.default_act = eval(act)
            if verbose:
                ultralytics_tasks.LOGGER.info(
                    f"{ultralytics_tasks.colorstr('activation:')} {act}"
                )

        if verbose:
            ultralytics_tasks.LOGGER.info(
                f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
            )
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else ultralytics_tasks.__dict__.get(m)
            )
            for j, a in enumerate(args):
                if isinstance(a, str):
                    try:
                        args[j] = (
                            locals()[a]
                            if a in locals()
                            else (
                                ultralytics_tasks.__dict__[a]
                                if a in ultralytics_tasks.__dict__
                                else ast.literal_eval(a)
                            )
                        )
                    except (ValueError, KeyError):
                        pass
            n = n_ = max(round(n * depth), 1) if n > 1 else n

            # Modules that take [c1, c2, *args[1:]] channels
            base_conv_modules = {
                ultralytics_tasks.Classify,
                ultralytics_tasks.Conv,
                ultralytics_tasks.ConvTranspose,
                ultralytics_tasks.GhostConv,
                ultralytics_tasks.Bottleneck,
                ultralytics_tasks.GhostBottleneck,
                ultralytics_tasks.SPP,
                ultralytics_tasks.SPPF,
                ultralytics_tasks.C2fPSA,
                ultralytics_tasks.C2PSA,
                ultralytics_tasks.DWConv,
                ultralytics_tasks.Focus,
                ultralytics_tasks.BottleneckCSP,
                ultralytics_tasks.C1,
                ultralytics_tasks.C2,
                ultralytics_tasks.C2f,
                ultralytics_tasks.C3k2,
                ultralytics_tasks.RepNCSPELAN4,
                ultralytics_tasks.ELAN1,
                ultralytics_tasks.ADown,
                ultralytics_tasks.AConv,
                ultralytics_tasks.SPPELAN,
                ultralytics_tasks.C2fAttn,
                ultralytics_tasks.C3,
                ultralytics_tasks.C3TR,
                ultralytics_tasks.C3Ghost,
                nn.ConvTranspose2d,
                ultralytics_tasks.DWConvTranspose2d,
                ultralytics_tasks.C3x,
                ultralytics_tasks.RepC3,
                ultralytics_tasks.PSA,
                ultralytics_tasks.SCDown,
                ultralytics_tasks.C2fCIB,
                RGG,  # Custom Paddy-YOLO module
            }

            repeat_arg_modules = {
                ultralytics_tasks.BottleneckCSP,
                ultralytics_tasks.C1,
                ultralytics_tasks.C2,
                ultralytics_tasks.C2f,
                ultralytics_tasks.C3k2,
                ultralytics_tasks.C2fAttn,
                ultralytics_tasks.C3,
                ultralytics_tasks.C3TR,
                ultralytics_tasks.C3Ghost,
                ultralytics_tasks.C3x,
                ultralytics_tasks.RepC3,
                ultralytics_tasks.C2fPSA,
                ultralytics_tasks.C2fCIB,
                ultralytics_tasks.C2PSA,
                RGG,  # Custom Paddy-YOLO module: args.insert(2, n); n = 1
            }

            if m in base_conv_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = ultralytics_tasks.make_divisible(
                        min(c2, max_channels) * width, 8
                    )
                if m is getattr(ultralytics_tasks, "C2fAttn", None):
                    args[1] = ultralytics_tasks.make_divisible(
                        min(args[1], max_channels // 2) * width, 8
                    )
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                        if args[2] > 1
                        else args[2]
                    )

                args = [c1, c2, *args[1:]]
                if m in repeat_arg_modules:
                    args.insert(2, n)
                    n = 1
                if m is getattr(ultralytics_tasks, "C3k2", None):
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
            elif m in {
                getattr(ultralytics_tasks, "AIFI", None),
                DySample,
                getattr(ultralytics_tasks, "DySample", None),
            }:
                args = [ch[f], *args]
                c2 = ch[f]
            elif m in {
                getattr(ultralytics_tasks, "HGStem", None),
                getattr(ultralytics_tasks, "HGBlock", None),
            }:
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is getattr(ultralytics_tasks, "HGBlock", None):
                    args.insert(4, n)
                    n = 1
            elif m is getattr(ultralytics_tasks, "ResNetLayer", None):
                c2 = args[1] if args[3] else args[1] * 4
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is ultralytics_tasks.Concat:
                c2 = sum(ch[x] for x in f)
            elif m in {
                ultralytics_tasks.Detect,
                getattr(ultralytics_tasks, "WorldDetect", None),
                getattr(ultralytics_tasks, "Segment", None),
                getattr(ultralytics_tasks, "Pose", None),
                getattr(ultralytics_tasks, "OBB", None),
                getattr(ultralytics_tasks, "ImagePoolingAttn", None),
                getattr(ultralytics_tasks, "v10Detect", None),
            }:
                args.append([ch[x] for x in f])
                if m is getattr(ultralytics_tasks, "Segment", None):
                    args[2] = ultralytics_tasks.make_divisible(
                        min(args[2], max_channels) * width, 8
                    )
                if m in {
                    ultralytics_tasks.Detect,
                    getattr(ultralytics_tasks, "Segment", None),
                    getattr(ultralytics_tasks, "Pose", None),
                    getattr(ultralytics_tasks, "OBB", None),
                }:
                    m.legacy = legacy
            elif m is getattr(ultralytics_tasks, "RTDETRDecoder", None):
                args.insert(1, [ch[x] for x in f])
            elif m is getattr(ultralytics_tasks, "CBLinear", None):
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is getattr(ultralytics_tasks, "CBFuse", None):
                c2 = ch[f[-1]]
            else:
                c2 = ch[f]

            m_ = (
                nn.Sequential(*(m(*args) for _ in range(n)))
                if n > 1
                else m(*args)
            )
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                ultralytics_tasks.LOGGER.info(
                    f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}"
                )
            save.extend(
                x % i for x in ([f] if isinstance(f, int) else f) if x != -1
            )
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    return fallback_parse_model


def register_paddy_modules() -> None:
    """Inject custom Paddy-YOLO modules into Ultralytics and update parse_model.

    This function:
    1. Registers RGG, RGGBottleneck, RGGConv, RepConv, GhostConv into:
       - `ultralytics.nn.modules`
       - `ultralytics.nn.modules.block`
       - `ultralytics.nn.tasks`
    2. Updates `__all__` lists in those modules if present.
    3. Patches `ultralytics.nn.tasks.parse_model` so `RGG` is parsed with the
       C2f argument convention `[c1, c2, *args[1:]]` and receives repeat count `n`.
    """
    global _PADDY_MODULES_REGISTERED

    # 1. Register modules in namespaces
    for name, cls in CUSTOM_MODULES.items():
        setattr(ultralytics_modules, name, cls)
        setattr(ultralytics_block, name, cls)
        setattr(ultralytics_tasks, name, cls)

        # Update __all__ lists
        if hasattr(ultralytics_modules, "__all__") and name not in ultralytics_modules.__all__:
            if isinstance(ultralytics_modules.__all__, list):
                ultralytics_modules.__all__.append(name)
            elif isinstance(ultralytics_modules.__all__, tuple):
                ultralytics_modules.__all__ = ultralytics_modules.__all__ + (name,)
        if hasattr(ultralytics_block, "__all__") and name not in ultralytics_block.__all__:
            if isinstance(ultralytics_block.__all__, list):
                ultralytics_block.__all__.append(name)
            elif isinstance(ultralytics_block.__all__, tuple):
                ultralytics_block.__all__ = ultralytics_block.__all__ + (name,)

    if _PADDY_MODULES_REGISTERED or getattr(ultralytics_tasks.parse_model, "_is_paddy_patched", False):
        _PADDY_MODULES_REGISTERED = True
        return

    # 2. Patch parse_model if not already patched
    patched = False
    try:
        src = inspect.getsource(ultralytics_tasks.parse_model)
        if "RGG" not in src and "C2f," in src:
            new_src = src.replace("C2f,", "C2f,\n            RGG,")
            if "m is AIFI:" in new_src:
                new_src = new_src.replace("m is AIFI:", "m in (AIFI, DySample):")
            local_ns: dict = {}
            exec(new_src, ultralytics_tasks.__dict__, local_ns)
            new_parse_model = local_ns["parse_model"]
            new_parse_model._is_paddy_patched = True
            ultralytics_tasks.parse_model = new_parse_model
            ultralytics_nn.parse_model = new_parse_model
            patched = True
        elif "RGG" in src and "DySample" not in src and "m is AIFI:" in src:
            new_src = src.replace("m is AIFI:", "m in (AIFI, DySample):")
            local_ns: dict = {}
            exec(new_src, ultralytics_tasks.__dict__, local_ns)
            new_parse_model = local_ns["parse_model"]
            new_parse_model._is_paddy_patched = True
            ultralytics_tasks.parse_model = new_parse_model
            ultralytics_nn.parse_model = new_parse_model
            patched = True
        elif "RGG" in src:
            ultralytics_tasks.parse_model._is_paddy_patched = True
            patched = True
    except Exception:
        patched = False

    if not patched:
        # Use robust fallback parse_model
        fallback_fn = _build_fallback_parse_model()
        fallback_fn._is_paddy_patched = True
        ultralytics_tasks.parse_model = fallback_fn
        ultralytics_nn.parse_model = fallback_fn

    # 3. Enhance guess_model_scale to default to 'n' for paddy-yolo models
    if hasattr(ultralytics_tasks, "guess_model_scale"):
        orig_guess_scale = ultralytics_tasks.guess_model_scale

        def paddy_guess_model_scale(model_path):
            scale = orig_guess_scale(model_path)
            if not scale and "paddy" in str(model_path).lower():
                return "n"
            return scale

        ultralytics_tasks.guess_model_scale = paddy_guess_model_scale
        if hasattr(ultralytics_nn, "guess_model_scale"):
            ultralytics_nn.guess_model_scale = paddy_guess_model_scale

    _PADDY_MODULES_REGISTERED = True
