#!/usr/bin/env python3
"""
Create a LaTeX system report for reproducibility.

"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OSInfo:
    os: str
    kernel: str
    distro: str
    hostname: str


@dataclass
class CPUInfo:
    model: str
    cores_logical: int
    cores_physical: Optional[int]


@dataclass
class RAMInfo:
    total_gb: float


@dataclass
class GPUInfo:
    name: str
    memory_total_mb: Optional[int]
    driver_version: Optional[str]
    cuda_version: Optional[str]


@dataclass
class CUDAToolkitInfo:
    present: bool
    nvcc_version: Optional[str]


@dataclass
class PythonInfo:
    version: str
    executable: str


@dataclass
class HFModelInfo:
    model_id: str
    snapshot_dir: Optional[str]
    local_path: Optional[str]


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def run_cmd(cmd: List[str]) -> Tuple[bool, str, str]:
    """Run a command and return (ok, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0]}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, "", f"Error running {cmd}: {exc}"


def latex_escape(text: str) -> str:
    """Escape LaTeX special chars in a simple but robust way."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def safe_version(pkg: str) -> str:
    """Return installed package version or 'n/a'."""
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "n/a"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Collect system information
# ---------------------------------------------------------------------------


def get_os_info() -> OSInfo:
    """Collect basic OS information."""
    os_name = platform.system() + " " + platform.release()
    kernel = platform.version()
    hostname = platform.node() or "unknown"

    distro = "unknown"
    os_release = Path("/etc/os-release")
    if os_release.exists():
        try:
            lines = os_release.read_text(encoding="utf-8").splitlines()
            data: Dict[str, str] = {}
            for line in lines:
                if "=" in line:
                    key, val = line.split("=", 1)
                    data[key.strip()] = val.strip().strip('"')
            pretty = data.get("PRETTY_NAME") or ""
            if pretty:
                distro = pretty
        except Exception:
            pass

    return OSInfo(os=os_name, kernel=kernel, distro=distro, hostname=hostname)


def get_cpu_info() -> CPUInfo:
    """Collect CPU model and core counts."""
    model = platform.processor() or "unknown"
    logical = os.cpu_count() or 0
    physical: Optional[int] = None

    # Try lscpu for better model name and physical cores
    ok, out, _ = run_cmd(["lscpu"])
    if ok and out:
        lines = out.splitlines()
        for line in lines:
            if line.startswith("Model name:"):
                model = line.split(":", 1)[1].strip()
            if line.startswith("CPU(s):") and logical == 0:
                try:
                    logical = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            if line.startswith("Core(s) per socket:"):
                try:
                    cores_per_socket = int(line.split(":", 1)[1].strip())
                except ValueError:
                    cores_per_socket = None
            elif line.startswith("Socket(s):"):
                try:
                    sockets = int(line.split(":", 1)[1].strip())
                except ValueError:
                    sockets = None
        # crude heuristic for physical cores
        try:
            cores_per_socket = next(
                int(line.split(":", 1)[1].strip())
                for line in lines
                if line.startswith("Core(s) per socket:")
            )
            sockets = next(
                int(line.split(":", 1)[1].strip())
                for line in lines
                if line.startswith("Socket(s):")
            )
            physical = cores_per_socket * sockets
        except Exception:
            physical = None

    return CPUInfo(model=model, cores_logical=logical, cores_physical=physical)


def get_ram_info() -> RAMInfo:
    """Collect total RAM in GB."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    kb = int(parts[1])
                    gb = round(kb / (1024**2), 2)  # kB -> GB
                    return RAMInfo(total_gb=gb)
        except Exception:
            pass
    # Fallback if /proc/meminfo is not available
    try:
        import psutil  # type: ignore

        gb = round(psutil.virtual_memory().total / (1024**3), 2)
        return RAMInfo(total_gb=gb)
    except Exception:
        return RAMInfo(total_gb=-1.0)


def get_gpu_info() -> List[GPUInfo]:
    """Liest GPU-Infos robust über `nvidia-smi -q --xml-format` aus."""
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "--xml-format"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0 or not result.stdout.strip():
        # nvidia-smi vorhanden, aber liefert Fehler → lieber still nichts melden,
        # statt "keine GPU" zu behaupten.
        return []

    try:
        root = ET.fromstring(result.stdout)
    except Exception:
        return []

    driver_version = root.findtext("driver_version")
    cuda_version = root.findtext("cuda_version")

    gpus: List[GPUInfo] = []
    for gpu_el in root.findall("gpu"):
        name = gpu_el.findtext("product_name")

        mem_total_el = gpu_el.find("fb_memory_usage/total")
        if mem_total_el is not None and mem_total_el.text:
            memory_total = mem_total_el.text.strip()
        else:
            memory_total = None

        gpus.append(
            GPUInfo(
                name=name,
                driver_version=driver_version,
                cuda_version=cuda_version,
                memory_total_mb=memory_total,
            )
        )
    return gpus


def get_cuda_toolkit_info() -> CUDAToolkitInfo:
    """Check nvcc and parse its version if present."""
    if shutil.which("nvcc") is None:
        return CUDAToolkitInfo(present=False, nvcc_version=None)

    ok, out, _ = run_cmd(["nvcc", "--version"])
    if not ok or not out:
        return CUDAToolkitInfo(present=False, nvcc_version=None)

    version_str: Optional[str] = None
    for line in out.splitlines():
        if "release" in line:
            # e.g. "Cuda compilation tools, release 12.6, V12.6.77"
            try:
                after = line.split("release", 1)[1].strip()
                version_str = after.split(",", 1)[0].strip()
            except Exception:
                pass
    return CUDAToolkitInfo(present=True, nvcc_version=version_str)


def get_python_info() -> PythonInfo:
    """Collect Python version and executable path."""
    return PythonInfo(version=sys.version.replace("\n", " "), executable=sys.executable)


def get_env_vars(keys: List[str]) -> Dict[str, str]:
    """Return a dict {key: value} for existing environment variables."""
    return {k: v for k, v in os.environ.items() if k in keys}


def get_package_versions(pkgs: List[str]) -> Dict[str, str]:
    """Return a dict {package: version} for selected packages."""
    return {pkg: safe_version(pkg) for pkg in pkgs}


def detect_model_snapshot(model_id: str, hf_home: Optional[str] = None) -> HFModelInfo:
    """
    Try to detect the local snapshot (revision) of a HuggingFace model.

    - Nutzt den Standard-HF-Cache (~/.cache/huggingface/hub/models--...).
    - Wählt den jüngsten Snapshot-Ordner nach mtime.
    """
    if hf_home is None:
        hf_home = os.environ.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )

    hub_dir = Path(hf_home) / "hub"
    # HF-Cache-Logik: models--ORG--NAME
    safe_name = model_id.replace("/", "--")
    model_dir = hub_dir / f"models--{safe_name}"

    if not model_dir.exists():
        return HFModelInfo(model_id=model_id, snapshot_dir=None, local_path=None)

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return HFModelInfo(model_id=model_id, snapshot_dir=None, local_path=None)

    snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        return HFModelInfo(model_id=model_id, snapshot_dir=None, local_path=None)

    # Wähle den Snapshot mit jüngster mtime
    snapshot_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best = snapshot_dirs[0]

    return HFModelInfo(
        model_id=model_id,
        snapshot_dir=best.name,
        local_path=str(best),
    )


# ---------------------------------------------------------------------------
# LaTeX writer
# ---------------------------------------------------------------------------


def write_latex(
    out_path: Path,
    os_info: OSInfo,
    cpu_info: CPUInfo,
    ram_info: RAMInfo,
    gpus: List[GPUInfo],
    cuda_toolkit: CUDAToolkitInfo,
    py_info: PythonInfo,
    env_vars: Dict[str, str],
    pkg_versions: Dict[str, str],
    hf_models: List[HFModelInfo],
) -> None:
    """Write the LaTeX report."""
    lines: List[str] = []
    lines.append("% Auto-generated system report")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")

    # --- Betriebssystem ---
    lines.append("\\subsubsection{Betriebssystem}")
    lines.append("\\begin{tabularx}{\\textwidth}{l X}")
    lines.append("\\toprule")
    lines.append("\\textbf{Parameter} & \\textbf{Wert} \\\\")
    lines.append("\\midrule")
    lines.append(f"OS & {latex_escape(os_info.os)} \\\\")
    lines.append(f"Kernel/Version & {latex_escape(os_info.kernel)} \\\\")
    lines.append(f"Distribution & {latex_escape(os_info.distro)} \\\\")
    lines.append(f"Hostname & {latex_escape(os_info.hostname)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("")

    # --- CPU & RAM ---
    lines.append("\\subsubsection{CPU und Arbeitsspeicher}")
    lines.append("\\begin{tabularx}{\\textwidth}{l X}")
    lines.append("\\toprule")
    lines.append("\\textbf{Parameter} & \\textbf{Wert} \\\\")
    lines.append("\\midrule")
    lines.append(f"CPU-Modell & {latex_escape(cpu_info.model)} \\\\")
    lines.append(f"Logische Kerne & {cpu_info.cores_logical} \\\\")
    if cpu_info.cores_physical is not None:
        lines.append(f"Physische Kerne & {cpu_info.cores_physical} \\\\")
    lines.append(f"Installierter RAM & {ram_info.total_gb:.2f}\\,GB \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("")

    # --- GPU ---
    lines.append("\\subsubsection{GPU-Konfiguration}")
    if not gpus:
        lines.append("Keine NVIDIA-GPU erkannt (\\texttt{nvidia-smi} nicht verfügbar).")
    else:
        lines.append("\\begin{tabularx}{\\textwidth}{l l l l}")
        lines.append("\\toprule")
        lines.append(
            "\\textbf{GPU} & \\textbf{Speicher} & "
            "\\textbf{Treiber} & \\textbf{CUDA-Version} \\\\"
        )
        lines.append("\\midrule")
        for gpu in gpus:
            mem_str = (
                f"{gpu.memory_total_mb} MiB"
                if gpu.memory_total_mb is not None
                else "n/a"
            )
            drv = gpu.driver_version or "n/a"
            cuda = gpu.cuda_version or "n/a"
            lines.append(
                f"{latex_escape(gpu.name)} & {mem_str} & {latex_escape(drv)} & "
                f"{latex_escape(cuda)} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabularx}")
    lines.append("")

    # --- CUDA Toolkit ---
    lines.append("\\subsubsection{CUDA-Toolkit}")
    lines.append("\\begin{tabularx}{\\textwidth}{l X}")
    lines.append("\\toprule")
    lines.append("\\textbf{Parameter} & \\textbf{Wert} \\\\")
    lines.append("\\midrule")
    if cuda_toolkit.present:
        ver = cuda_toolkit.nvcc_version or "unbekannt"
        lines.append(f"nvcc & Version {latex_escape(ver)} \\\\")
    else:
        lines.append("nvcc & nicht installiert \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("")

    # --- Python ---
    lines.append("\\subsubsection{Python-Umgebung}")
    lines.append("\\begin{tabularx}{\\textwidth}{l X}")
    lines.append("\\toprule")
    lines.append("\\textbf{Parameter} & \\textbf{Wert} \\\\")
    lines.append("\\midrule")
    lines.append(f"Python-Version & {latex_escape(py_info.version)} \\\\")
    lines.append(f"Python-Executable & {latex_escape(py_info.executable)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("")

    # --- Env Vars ---
    if env_vars:
        lines.append("\\subsubsection{Wichtige Umgebungsvariablen}")
        lines.append("\\begin{tabularx}{\\textwidth}{l X}")
        lines.append("\\toprule")
        lines.append("\\textbf{Variable} & \\textbf{Wert} \\\\")
        lines.append("\\midrule")
        for k, v in sorted(env_vars.items()):
            lines.append(f"{latex_escape(k)} & {latex_escape(v)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabularx}")
        lines.append("")

    # --- Packages ---
    lines.append("\\subsubsection{Verwendete Python-Pakete}")
    lines.append("\\begin{scriptsize}")
    lines.append("\\begin{longtable}{l l}")
    lines.append("\\toprule")
    lines.append("\\textbf{Paket} & \\textbf{Version} \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append("\\textbf{Paket} & \\textbf{Version} \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    for pkg, ver in sorted(pkg_versions.items()):
        lines.append(f"{latex_escape(pkg)} & {latex_escape(ver)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    lines.append("\\end{scriptsize}")
    lines.append("")

    # --- HF Models ---
    lines.append("\\subsubsection{Verwendete Embedding-Modelle}")
    if not hf_models:
        lines.append("Keine lokalen Snapshots im HuggingFace-Cache gefunden.")
    else:
        lines.append("\\begin{tabularx}{\\textwidth}{l l X}")
        lines.append("\\toprule")
        lines.append(
            "\\textbf{Model-ID} & \\textbf{Snapshot/Revision} & "
            "\\textbf{Lokaler Pfad} \\\\"
        )
        lines.append("\\midrule")
        for m in hf_models:
            snap = m.snapshot_dir or "n/a"
            path = m.local_path or "n/a"
            lines.append(f"{latex_escape(m.model_id)} & {latex_escape(snap)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabularx}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os_info = get_os_info()
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    gpus = get_gpu_info()
    cuda_toolkit = get_cuda_toolkit_info()
    py_info = get_python_info()

    env_keys = [
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "MAX_TOKENS_PER_BATCH",
    ]
    env_vars = get_env_vars(env_keys)

    # Wichtige Pakete – nach Bedarf erweiterbar
    key_pkgs = [
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "tqdm",
        "huggingface-hub",
    ]
    pkg_versions = get_package_versions(key_pkgs)

    # Verwendete Embedding-Modelle
    hf_models = [
        detect_model_snapshot("Qwen/Qwen3-Embedding-0.6B"),
        detect_model_snapshot("jinaai/jina-embeddings-v2-small-en"),
    ]

    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "src" / "code_output" / "system_report.tex"

    write_latex(
        out_path=out_path,
        os_info=os_info,
        cpu_info=cpu_info,
        ram_info=ram_info,
        gpus=gpus,
        cuda_toolkit=cuda_toolkit,
        py_info=py_info,
        env_vars=env_vars,
        pkg_versions=pkg_versions,
        hf_models=hf_models,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
