"""Build configuration for the optional compiled C++ kernel (snn_opt._kernel).

All project metadata lives in pyproject.toml; this file exists only to declare
the pybind11 extension module. The extension is marked ``optional`` -- a plain
``pip install`` falls back to the pure-Python solver when no C++ compiler is
available, and ``backend='c'`` then raises a clear error at solve time.

OpenMP: the kernel parallelises its matvec across a thread team when built with
full ``-fopenmp`` (the ``backend='c_openmp'`` / multicore-``'c'`` path). OpenMP
is not universally available (notably stock AppleClang), so we *probe* the
compiler at build time and add ``-fopenmp`` only when it compiles+links. When it
does not, the extension still builds SIMD-only (``-fopenmp-simd``) -- the
``'c'`` and ``'c_serial'`` backends keep working and ``'c_openmp'`` raises a
clear error. The C++ side keys off the compiler-defined ``_OPENMP`` macro, so no
custom define is needed.

Build in place for development with::

    python setup.py build_ext --inplace
"""

import contextlib
import os
import sys
import tempfile

from setuptools import setup


def _openmp_works():
    """True if the C compiler can build *and link* a trivial -fopenmp program.

    Probes with a fresh compiler instance on a throwaway source file. Any
    failure (unsupported flag, missing libgomp/libomp, link error) returns
    False, which downgrades the build to SIMD-only -- never fatal.
    """
    try:
        from setuptools._distutils.ccompiler import new_compiler
        from setuptools._distutils.sysconfig import customize_compiler
    except Exception:  # pragma: no cover - very old setuptools
        try:
            from distutils.ccompiler import new_compiler
            from distutils.sysconfig import customize_compiler
        except Exception:
            return False

    src = "#include <omp.h>\nint main(){return omp_get_num_threads();}\n"
    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "omp_probe.c")
        with open(src_path, "w") as fh:
            fh.write(src)
        cc = new_compiler()
        with contextlib.suppress(Exception):
            customize_compiler(cc)
        devnull = open(os.devnull, "w")
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 2)  # silence probe compiler chatter
            objs = cc.compile([src_path], output_dir=tmp,
                              extra_postargs=["-fopenmp"])
            cc.link_executable(objs, os.path.join(tmp, "omp_probe"),
                               extra_postargs=["-fopenmp"])
        except Exception:
            return False
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            devnull.close()
    return True


try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:  # pybind11 not installed -- ship pure-Python only
    ext_modules = []
    cmdclass = {}
else:
    # Base flags: -fopenmp-simd vectorises the matvec sum-reduction without
    # pulling in the OpenMP runtime. If full OpenMP is available we swap in
    # -fopenmp (a superset that also enables the simd hints) and link it, which
    # defines _OPENMP and turns on the multicore matvec path.
    compile_args = ["-O3", "-march=native", "-funroll-loops"]
    link_args = []
    if _openmp_works():
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    else:
        compile_args.append("-fopenmp-simd")

    ext_modules = [
        Pybind11Extension(
            "snn_opt._kernel",
            ["src/snn_opt/_native/bindings.cpp"],
            # Track the header so editing the kernel triggers a rebuild
            # (setuptools only watches listed sources otherwise).
            depends=["src/snn_opt/_native/snn_qp_core.hpp"],
            cxx_std=17,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=True,
        ),
    ]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
