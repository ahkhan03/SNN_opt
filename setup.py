"""Build configuration for the optional compiled C++ kernel (snn_opt._kernel).

All project metadata lives in pyproject.toml; this file exists only to declare
the pybind11 extension module. The extension is marked ``optional`` -- a plain
``pip install`` falls back to the pure-Python solver when no C++ compiler is
available, and ``backend='c'`` then raises a clear error at solve time.

Build in place for development with::

    python setup.py build_ext --inplace
"""

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:  # pybind11 not installed -- ship pure-Python only
    ext_modules = []
    cmdclass = {}
else:
    ext_modules = [
        Pybind11Extension(
            "snn_opt._kernel",
            ["src/snn_opt/_native/bindings.cpp"],
            # Track the header so editing the kernel triggers a rebuild
            # (setuptools only watches listed sources otherwise).
            depends=["src/snn_opt/_native/snn_qp_core.hpp"],
            cxx_std=17,
            # -fopenmp-simd enables the `#pragma omp simd` reduction hints
            # (vectorised matvec) without pulling in the OpenMP runtime.
            extra_compile_args=["-O3", "-march=native", "-funroll-loops",
                                "-fopenmp-simd"],
            optional=True,
        ),
    ]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
