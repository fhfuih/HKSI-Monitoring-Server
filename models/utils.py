import os
import sys
import traceback
from datetime import datetime

import torch


## {{{ debug related
def datetime_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000)


## }}}

## {{{ torch devices
try:
    if torch.backends.mps.is_available():  # macOS metal
        GPU = torch.device("mps")
        _gpu_module = torch.mps

        # some ops are not available in macOS metal accelerator
        # This allows fallback to CPU
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    elif torch.cuda.is_available():  # NVidia CUDA
        GPU = torch.device("cuda")
        _gpu_module = torch.cuda

    elif torch.xpu.is_available():  # Intel CPU/XPU
        GPU = torch.device("xpu")
        _gpu_module = torch.xpu

    else:  # Regular CPU
        GPU = torch.device("cpu")
        _gpu_module = torch.cpu
except:
    print("Warning: exception occurs when detecting GPU devices:", file=sys.stderr)
    traceback.print_exc()
    print("==========", "Using CPU.", sep="\n", file=sys.stderr)
    GPU = torch.device("cpu")
    _gpu_module = torch.cpu


def empty_cache():
    # CPU doesn't have empty_cache
    _empty_cache = getattr(_gpu_module, "empty_cache", None)
    if _empty_cache is not None:
        _empty_cache()


## }}}
