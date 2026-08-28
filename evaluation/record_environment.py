"""
Captures the exact environment the experiments ran in, for the paper's
"Experimental Setup" / reproducibility table (reviewer requirement #12).

Usage:
    python -m evaluation.record_environment
"""
import json
import platform
import subprocess
import sys

from . import common  # noqa: E402
from .common import results_path


def get_pip_freeze():
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return sorted(out.strip().splitlines())
    except Exception as e:
        return [f"<pip freeze failed: {e}>"]


def main():
    import torch

    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["cuda_runtime_version"] = torch.version.cuda
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    else:
        info["cuda_device_name"] = None

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass

    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
        info["nvidia_driver_version"] = nvidia_smi
    except Exception:
        info["nvidia_driver_version"] = None

    info["pip_freeze"] = get_pip_freeze()

    out_path = results_path("experimental_setup.json")
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[env] wrote {out_path}")
    for k, v in info.items():
        if k != "pip_freeze":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
