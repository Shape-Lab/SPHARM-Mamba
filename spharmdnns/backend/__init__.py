from .array import take_along_dim, hstack, vstack, stack, unbind
from .linalg import inner, norm, normalize, cross
from .creation import full_like, one_hot
from .misc import clamp_min, sqrt


class ops:
    take_along_dim = take_along_dim
    hstack = hstack
    vstack = vstack
    stack = stack
    unbind = unbind
    inner = inner
    norm = norm
    normalize = normalize
    cross = cross
    full_like = full_like
    one_hot = one_hot
    clamp_min = clamp_min
    sqrt = sqrt


__all__ = ["ops"]
