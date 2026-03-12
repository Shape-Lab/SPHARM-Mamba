import sys, inspect, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mamba")))


import torch
import mamba_ssm
from mamba_ssm.modules.mamba_simple import Mamba

print("Loaded mamba_ssm from:", os.path.dirname(inspect.getfile(mamba_ssm)))