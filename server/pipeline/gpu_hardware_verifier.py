"""
gpu_hardware_verifier.py — Exact GPU Hardware Architecture & Telemetry Verifier

Provides ground-truth hardware discovery for NVIDIA GPUs on RunPod/Cloud instances:
1. Direct query of nvidia-smi (product name, driver version, VRAM total, GPU UUID).
2. Direct query of PyTorch CUDA runtime:
   - torch.cuda.get_device_name(0)
   - torch.cuda.get_device_capability(0) -> Compute Capability (sm_XX)
   - torch.cuda.mem_get_info() -> Free / Total VRAM
3. Architecture inference:
   - (8, 9): Ada Lovelace (RTX 4090, L4, L40, L40S) -> 4th-gen Tensor Cores + FP8
   - (8, 6): Ampere Client/Workstation (RTX 3090, RTX A5000, A4000) -> 3rd-gen Tensor Cores
   - (8, 0): Ampere Datacenter (A100) -> 3rd-gen Tensor Cores + TF32
   - (7, 5): Turing (T4, RTX 2080 Ti) -> 2nd-gen Tensor Cores
4. Recommended batch sizes based on physical architecture and VRAM headroom.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)


class GPUHardwareVerifier:
    """Discovers and validates exact physical GPU hardware specifications."""

    def __init__(self) -> None:
        self._cached_spec: Optional[Dict[str, Any]] = None

    def query_hardware(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Inspects system hardware via PyTorch, nvidia-smi, and system memory.
        Returns structured telemetry dictionary.
        """
        if self._cached_spec is not None and not force_refresh:
            return self._cached_spec

        spec: Dict[str, Any] = {
            "is_cuda_available": False,
            "is_mps_available": False,
            "device_name": "CPU",
            "exact_gpu_model": "Unknown",
            "compute_capability": None,
            "architecture": "CPU",
            "driver_version": None,
            "total_vram_gb": 0.0,
            "free_vram_gb": 0.0,
            "system_ram_gb": 0.0,
            "tensor_cores_gen": None,
            "recommended_yolo_batch": 4,
            "recommended_samurai_workers": 2,
            "detection_source": "fallback",
        }

        # 1. System Memory via psutil or os
        try:
            import psutil
            vm = psutil.virtual_memory()
            spec["system_ram_gb"] = round(vm.total / (1024.0 ** 3), 2)
        except ImportError:
            try:
                # POSIX fallback
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                spec["system_ram_gb"] = round((pages * page_size) / (1024.0 ** 3), 2)
            except Exception:
                pass

        # 2. PyTorch CUDA inspection
        try:
            import torch
            if torch.cuda.is_available():
                spec["is_cuda_available"] = True
                spec["device_name"] = torch.cuda.get_device_name(0)
                spec["exact_gpu_model"] = spec["device_name"]
                cap = torch.cuda.get_device_capability(0)
                spec["compute_capability"] = f"{cap[0]}.{cap[1]}"

                # Map compute capability to architecture
                arch_map = {
                    (9, 0): ("Hopper", "5th Gen (FP8/Transformer Engine)"),
                    (8, 9): ("Ada Lovelace", "4th Gen (FP8/Hopper Tensor)"),
                    (8, 6): ("Ampere (Client/Pro)", "3rd Gen (TF32/Bfloat16)"),
                    (8, 0): ("Ampere (Datacenter)", "3rd Gen (TF32/Bfloat16)"),
                    (7, 5): ("Turing", "2nd Gen (FP16/INT8)"),
                    (7, 0): ("Volta", "1st Gen (FP16)"),
                }
                arch_info = arch_map.get(cap, (f"CUDA sm_{cap[0]}{cap[1]}", "Standard Tensor"))
                spec["architecture"] = arch_info[0]
                spec["tensor_cores_gen"] = arch_info[1]

                free_bytes, total_bytes = torch.cuda.mem_get_info()
                spec["free_vram_gb"] = round(free_bytes / (1024.0 ** 3), 2)
                spec["total_vram_gb"] = round(total_bytes / (1024.0 ** 3), 2)
                spec["detection_source"] = "torch.cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                spec["is_mps_available"] = True
                spec["device_name"] = "Apple Silicon (MPS)"
                spec["exact_gpu_model"] = "Apple Silicon Unified Memory GPU"
                spec["architecture"] = "Apple Silicon"
                spec["detection_source"] = "torch.mps"
        except ImportError:
            pass

        # 3. Direct nvidia-smi execution for ground truth confirmation
        if shutil.which("nvidia-smi"):
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and res.stdout.strip():
                    parts = [p.strip() for p in res.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        spec["exact_gpu_model"] = parts[0]
                        spec["driver_version"] = parts[1]
                        if not spec["total_vram_gb"]:
                            spec["total_vram_gb"] = round(float(parts[2]) / 1024.0, 2)
                        spec["detection_source"] = "nvidia-smi + torch"
            except Exception as e:
                log.debug("nvidia-smi query failed: %s", e)

        # 4. Compute hardware-informed recommendations
        total_vram = spec["total_vram_gb"]
        sys_ram = spec["system_ram_gb"]

        if total_vram >= 40:  # A100 (40GB/80GB)
            spec["recommended_yolo_batch"] = 128
            spec["recommended_samurai_workers"] = 8 if sys_ram >= 64 else 4
        elif total_vram >= 22:  # RTX 3090 / 4090 / A5000 / L40S (24GB)
            spec["recommended_yolo_batch"] = 64
            spec["recommended_samurai_workers"] = 4 if sys_ram >= 32 else 2
        elif total_vram >= 14:  # T4 / RTX 4000 (16GB)
            spec["recommended_yolo_batch"] = 32
            spec["recommended_samurai_workers"] = 2
        else:
            spec["recommended_yolo_batch"] = 16 if spec["is_cuda_available"] else 4
            spec["recommended_samurai_workers"] = 1

        self._cached_spec = spec
        return spec

    def format_diagnostic_report(self) -> str:
        """Generates a human-readable diagnostic hardware summary."""
        s = self.query_hardware()
        lines = [
            "=" * 60,
            "  GPU HARDWARE & ACCELERATION SPECIFICATION REPORT",
            "=" * 60,
            f"  Exact GPU Model     : {s['exact_gpu_model']}",
            f"  Architecture        : {s['architecture']} ({s['compute_capability'] or 'N/A'})",
            f"  Tensor Cores Gen    : {s['tensor_cores_gen'] or 'N/A'}",
            f"  Driver Version      : {s['driver_version'] or 'N/A'}",
            f"  VRAM (Free / Total) : {s['free_vram_gb']} GB / {s['total_vram_gb']} GB",
            f"  Host System RAM     : {s['system_ram_gb']} GB",
            f"  Telemetry Source    : {s['detection_source']}",
            "-" * 60,
            f"  Recommended Batch   : {s['recommended_yolo_batch']} frames/batch",
            f"  Max Safe SAMURAI    : {s['recommended_samurai_workers']} workers (memory bounded)",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    verifier = GPUHardwareVerifier()
    print(verifier.format_diagnostic_report())
