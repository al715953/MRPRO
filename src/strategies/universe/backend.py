import platform
import numpy as np


class UniverseBackend:
    """Selector de Backend V13.9.4: Gestión de Aceleración por Hardware."""

    @staticmethod
    def get_xp():
        # 1. Intento de activación de NVIDIA CUDA
        if platform.system() != "Darwin":
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() > 0:
                    return cp, "NVIDIA (CuPy/CUDA)"
            except:
                pass

        # 2. Identificación de Entorno Apple o CPU Estándar
        is_apple = platform.system() == "Darwin" or platform.machine().startswith("arm")
        backend_name = (
            "Apple Silicon (NumPy/AMX)" if is_apple else "Standard CPU (NumPy)"
        )
        return np, backend_name
