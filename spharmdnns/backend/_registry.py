"""
July 2025

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
import torch

_backend_registry = {}


def _infer_backend(*args):
    def flatten(items):
        for item in items:
            if isinstance(item, dict):
                yield from flatten(item.values())
            elif isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    for arg in flatten(args):
        if isinstance(arg, np.ndarray):
            return "numpy"
        elif isinstance(arg, torch.Tensor):
            return "torch"
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")


def register(name):
    """
    Decorator to register a backend implementation.
    """

    def decorator(fn):
        _backend_registry.setdefault(fn.__name__, {})[name] = fn

        def wrapper(*args, **kwargs):
            backend = kwargs.pop("backend", None)
            if backend is None:
                backend = _infer_backend(*args, *kwargs.values())

            impl = _backend_registry.get(fn.__name__, {}).get(backend)
            if impl is None:
                raise RuntimeError(f"{fn.__name__} not implemented for backend '{backend}'")
            return impl(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator
