import mmap
import numpy as np

def create_page_aligned_tensor(shape, dtype=np.float32):
    """
    Allocates strictly 4096-byte aligned memory for UMA Zero-Copy.
    Returns: (numpy_array_view, mmap_object_reference)
    """
    size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    alloc_size = (size_bytes + 4095) & ~4095
    
    mm = mmap.mmap(-1, alloc_size, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    arr = np.frombuffer(mm, dtype=dtype, count=np.prod(shape)).reshape(shape)
    
    return arr, mm
