import os
# Configure OpenBLAS/MKL thread limits before any numpy/torch imports to prevent multi-process memory allocation failures on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from neon_arena.cli import main

if __name__ == "__main__":
    main()
