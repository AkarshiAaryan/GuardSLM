import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict

def collect_run_metadata(
    model_name: str,
    checkpoint: str = None,
    dataset_path: str = None,
    generation_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Collects system, model, and dataset metadata for reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model_name": model_name,
        "checkpoint": checkpoint,
        "dataset_path": dataset_path,
        "generation_params": generation_params or {},
        "cuda_available": False,
        "gpu_name": None
    }

    try:
        import torch
        metadata["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            metadata["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return metadata
