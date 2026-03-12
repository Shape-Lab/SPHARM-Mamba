import sys
from importlib.metadata import version

try:
    __version__ = version("spharmdnns")
except ModuleNotFoundError:
    pass

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is unavailable on this system, which is required for SPHARM-DNNs backend. Please visit https://pytorch.org/get-started/."
    )

from .core import SPHARMNet, SPHARMReg 

# backward compatibility
SPHARM_Net = SPHARMNet
SPHARM_Reg = SPHARMReg
sys.modules["spharmdnns.lib"] = sys.modules["spharmdnns"]
