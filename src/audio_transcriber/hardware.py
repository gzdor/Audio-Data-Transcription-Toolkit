#!/usr/bin/env python3
"""
Hardware discovery and auto-tuning for multi-core CPUs, NVIDIA CUDA GPUs,
and AMD ROCm / HIP / DirectML GPUs.
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
import ctranslate2


@dataclass
class GPUDeviceInfo:
    index: int
    vendor: str  # "NVIDIA" | "AMD" | "OTHER"
    name: str
    backend: str  # "cuda" | "rocm" | "directml" | "unconfigured"
    vram_mb: Optional[int] = None
    compute_capability: Optional[str] = None
    supported_compute_types: List[str] = field(default_factory=list)


@dataclass
class HardwareProfile:
    os_name: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    gpus: List[GPUDeviceInfo] = field(default_factory=list)
    recommended_device: str = "cpu"          # "cuda" | "rocm" | "directml" | "cpu"
    recommended_backend: str = "ctranslate2" # "ctranslate2" | "pytorch_gpu"
    recommended_compute_type: str = "int8"
    recommended_cpu_threads: int = 4
    recommended_num_workers: int = 1
    recommended_batch_size: int = 16
    notes: List[str] = field(default_factory=list)

    def summary_line(self) -> str:
        gpu_str = (
            ", ".join(f"{g.vendor} {g.name} ({g.backend})" for g in self.gpus)
            if self.gpus
            else "None detected"
        )
        return (
            f"CPU: {self.cpu_model} ({self.physical_cores}P/{self.logical_cores}L cores) | "
            f"GPUs: {gpu_str} | "
            f"Recommended: device={self.recommended_device} ({self.recommended_backend}), "
            f"compute_type={self.recommended_compute_type}, "
            f"workers={self.recommended_num_workers}, threads/worker={self.recommended_cpu_threads}, "
            f"batch_size={self.recommended_batch_size}"
        )

    def to_dict(self) -> Dict:
        return asdict(self)


def _detect_cpu_topology() -> Tuple[str, int, int]:
    """Return (cpu_model_name, physical_cores, logical_cores)."""
    logical = os.cpu_count() or 4
    physical = 0
    model_name = platform.processor() or "Generic CPU"

    if os.path.exists("/proc/cpuinfo"):
        try:
            phys_core_pairs: Set[Tuple[str, str]] = set()
            cpu_cores_field: Optional[int] = None
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                phys_id = "0"
                core_id = None
                for line in f:
                    if ":" not in line:
                        continue
                    k, v = [p.strip() for p in line.split(":", 1)]
                    if k == "model name" and v:
                        model_name = v
                    elif k == "physical id":
                        phys_id = v
                    elif k == "core id":
                        core_id = v
                        phys_core_pairs.add((phys_id, core_id))
                    elif k == "cpu cores" and v.isdigit():
                        cpu_cores_field = int(v)
            if phys_core_pairs:
                physical = len(phys_core_pairs)
            elif cpu_cores_field:
                physical = cpu_cores_field
        except Exception:
            pass

    if physical <= 0:
        physical = max(1, logical // 2) if logical > 2 else logical

    return model_name, physical, logical


def _probe_nvidia_gpus() -> List[GPUDeviceInfo]:
    """Detect NVIDIA GPUs via CTranslate2, PyTorch CUDA, and nvidia-smi."""
    gpus: List[GPUDeviceInfo] = []
    ct2_cuda_count = 0
    try:
        ct2_cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        ct2_cuda_count = 0

    # Check if PyTorch is using AMD ROCm HIP rather than NVIDIA CUDA
    is_rocm_hip = False
    try:
        import torch
        if getattr(torch.version, "hip", None) is not None:
            is_rocm_hip = True
    except Exception:
        pass

    if is_rocm_hip:
        return []

    smi_rows: List[Tuple[str, Optional[int], Optional[str]]] = []
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                name = parts[0] if parts else "NVIDIA GPU"
                vram = int(float(parts[1])) if len(parts) > 1 and parts[1].replace(".", "").isdigit() else None
                cc = parts[2] if len(parts) > 2 else None
                smi_rows.append((name, vram, cc))
    except Exception:
        pass

    count = max(ct2_cuda_count, len(smi_rows))
    for idx in range(count):
        name, vram, cc = smi_rows[idx] if idx < len(smi_rows) else (f"NVIDIA CUDA GPU #{idx}", None, None)
        supported: List[str] = []
        if idx < ct2_cuda_count:
            try:
                supported = sorted(list(ctranslate2.get_supported_compute_types("cuda", idx)))
            except Exception:
                supported = ["float16", "int8_float16", "int8", "float32"]
        backend = "cuda" if idx < ct2_cuda_count else "unconfigured"
        gpus.append(
            GPUDeviceInfo(
                index=idx,
                vendor="NVIDIA",
                name=name,
                backend=backend,
                vram_mb=vram,
                compute_capability=cc,
                supported_compute_types=supported,
            )
        )
    return gpus


def _probe_amd_gpus() -> List[GPUDeviceInfo]:
    """Detect AMD GPUs via ROCm (/dev/kfd, rocm-smi, torch.version.hip) or DirectML."""
    gpus: List[GPUDeviceInfo] = []

    # 1. Check PyTorch ROCm (HIP)
    torch_rocm_available = False
    torch_rocm_names: List[str] = []
    try:
        import torch
        if getattr(torch.version, "hip", None) is not None and torch.cuda.is_available():
            torch_rocm_available = True
            for i in range(torch.cuda.device_count()):
                torch_rocm_names.append(torch.cuda.get_device_name(i))
    except Exception:
        pass

    # 2. Check CTranslate2 compiled with ROCm/HIP
    ct2_rocm_count = 0
    if torch_rocm_available or os.path.exists("/dev/kfd"):
        try:
            ct2_rocm_count = ctranslate2.get_cuda_device_count()
        except Exception:
            ct2_rocm_count = 0

    # 3. Check rocm-smi or /sys/class/kfd/kfd/topology/nodes
    rocm_smi_names: List[str] = []
    try:
        res = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().splitlines()[1:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1]:
                    rocm_smi_names.append(parts[1])
    except Exception:
        pass

    # 4. Check DirectML (torch_directml or onnxruntime DirectML on Windows/WSL)
    dml_available = False
    dml_name = "AMD DirectML GPU"
    try:
        import torch_directml
        if torch_directml.is_available():
            dml_available = True
            dml_name = torch_directml.device_name(0)
    except Exception:
        pass

    count = max(len(torch_rocm_names), ct2_rocm_count, len(rocm_smi_names))
    if count > 0:
        for idx in range(count):
            name = (
                torch_rocm_names[idx]
                if idx < len(torch_rocm_names)
                else (rocm_smi_names[idx] if idx < len(rocm_smi_names) else f"AMD Instinct/Radeon GPU #{idx}")
            )
            if idx < ct2_rocm_count:
                backend = "rocm"
                supported = ["float16", "int8_float16", "int8", "float32"]
            elif torch_rocm_available:
                backend = "rocm"
                supported = ["float16", "float32"]
            else:
                backend = "unconfigured"
                supported = []
            gpus.append(
                GPUDeviceInfo(
                    index=idx,
                    vendor="AMD",
                    name=name,
                    backend=backend,
                    supported_compute_types=supported,
                )
            )
    elif dml_available:
        gpus.append(
            GPUDeviceInfo(
                index=0,
                vendor="AMD",
                name=dml_name,
                backend="directml",
                supported_compute_types=["float16", "float32"],
            )
        )

    return gpus


def detect_hardware_profile(
    preferred_device: str = "auto",
    preferred_compute_type: str = "auto",
    user_threads: Optional[int] = None,
    user_workers: Optional[int] = None,
    user_batch_size: Optional[int] = None,
) -> HardwareProfile:
    """
    Auto-detect CPU and GPU (NVIDIA CUDA / AMD ROCm / DirectML) resources and compute
    optimal threads, workers, compute_type, and batch_size for scalable transcription.
    """
    cpu_model, phys_cores, log_cores = _detect_cpu_topology()
    nvidia_gpus = _probe_nvidia_gpus()
    amd_gpus = _probe_amd_gpus()
    all_gpus = nvidia_gpus + amd_gpus
    notes: List[str] = []

    req_dev = (preferred_device or "auto").lower()

    # Determine active device and backend
    device = "cpu"
    backend = "ctranslate2"
    compute_type = "int8"
    batch_size = 16

    active_nvidia = [g for g in nvidia_gpus if g.backend == "cuda"]
    active_amd = [g for g in amd_gpus if g.backend in ("rocm", "directml")]

    if req_dev in ("auto", "cuda", "gpu", "nvidia") and active_nvidia:
        g0 = active_nvidia[0]
        device = "cuda"
        backend = "ctranslate2"
        if "int8_float16" in g0.supported_compute_types:
            compute_type = "int8_float16"
        elif "float16" in g0.supported_compute_types:
            compute_type = "float16"
        else:
            compute_type = "int8"
        batch_size = 24
        notes.append(f"Using NVIDIA CUDA acceleration on {g0.name} via CTranslate2.")
    elif req_dev in ("auto", "rocm", "amd", "directml", "gpu") and active_amd:
        g0 = active_amd[0]
        device = g0.backend
        # Check if CTranslate2 itself was compiled with ROCm support
        ct2_count = 0
        try:
            ct2_count = ctranslate2.get_cuda_device_count()
        except Exception:
            ct2_count = 0
        if ct2_count > 0 and g0.backend == "rocm":
            backend = "ctranslate2"
            compute_type = "float16"
            batch_size = 24
            notes.append(f"Using AMD ROCm acceleration on {g0.name} via ROCm-compiled CTranslate2.")
        else:
            backend = "pytorch_gpu"
            compute_type = "float16"
            batch_size = 16
            notes.append(f"Using AMD GPU acceleration on {g0.name} via PyTorch ({g0.backend}).")
    else:
        if req_dev not in ("auto", "cpu"):
            notes.append(
                f"Requested device '{preferred_device}' is not available in current runtime; falling back to optimized multi-core CPU."
            )
        for g in all_gpus:
            if g.backend == "unconfigured":
                notes.append(
                    f"Detected {g.vendor} GPU ({g.name}), but GPU runtime (CUDA/ROCm CTranslate2 or PyTorch) is not active; using CPU."
                )
        device = "cpu"
        backend = "ctranslate2"
        cpu_supported = set()
        try:
            cpu_supported = set(ctranslate2.get_supported_compute_types("cpu"))
        except Exception:
            cpu_supported = {"int8", "float32"}
        compute_type = "int8" if "int8" in cpu_supported else "float32"
        batch_size = 16

    if preferred_compute_type and preferred_compute_type.lower() != "auto":
        compute_type = preferred_compute_type.lower()

    # Optimal CPU thread and worker partitioning:
    # Avoid OpenMP barrier contention by capping intra-op threads per worker at 4..8
    # and scaling inter-op workers across remaining physical cores.
    if user_workers is not None and user_workers > 0:
        num_workers = user_workers
    else:
        if device != "cpu":
            num_workers = max(1, len(active_nvidia) or len(active_amd))
        else:
            if phys_cores >= 8:
                num_workers = max(2, phys_cores // 4)
            elif phys_cores >= 4:
                num_workers = 2
            else:
                num_workers = 1

    if user_threads is not None and user_threads > 0:
        cpu_threads = user_threads
    else:
        if device != "cpu":
            cpu_threads = min(4, max(2, phys_cores))
        else:
            # Split physical cores across workers (minimum 2, up to 8 per worker)
            cpu_threads = max(2, min(8, phys_cores // max(1, num_workers)))
            # If running a single worker on a 4-core CPU, use all 4 cores
            if num_workers == 1:
                cpu_threads = max(2, min(8, phys_cores))

    if user_batch_size is not None and user_batch_size > 0:
        batch_size = user_batch_size

    return HardwareProfile(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_model=cpu_model,
        physical_cores=phys_cores,
        logical_cores=log_cores,
        gpus=all_gpus,
        recommended_device=device,
        recommended_backend=backend,
        recommended_compute_type=compute_type,
        recommended_cpu_threads=cpu_threads,
        recommended_num_workers=num_workers,
        recommended_batch_size=batch_size,
        notes=notes,
    )

