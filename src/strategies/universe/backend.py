import platform
import os
import numpy as np

class UniverseBackend:
    @staticmethod
    def get_xp():
        has_gpu_nvidia = False
        try:
            if platform.system() != 'Darwin':
                import cupy as cp
                has_gpu_nvidia = True
        except ImportError:
            pass
            
        xp = cp if has_gpu_nvidia else np
        is_apple = platform.processor() == 'arm' or platform.system() == 'Darwin'
        
        backend_name = "NVIDIA (CuPy)" if has_gpu_nvidia else "Apple M4 (NumPy/AMX)" if is_apple else "Standard CPU"
        return xp, backend_name