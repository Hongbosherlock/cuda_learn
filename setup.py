from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(
            name='cuda_reduce',
            sources=[
                'pyblind.cpp',
                'kernel/reduce_sum.cu',
            ],
            extra_compile_args={
                'nvcc': [
                    '-DNDEBUG',
                    '-O3',
                    '-g',
                    '-lineinfo',
                    '--keep', 
                    '--ptxas-options=--warn-on-local-memory-usage',
                    '--ptxas-options=--warn-on-spills',
                    '--resource-usage',
                    '--source-in-ptx',
                    '--use_fast_math',
                    '-gencode=arch=compute_90a, code=sm_90a',
                ]
            },
            include_dirs=[
                'kernel',
                '/usr/local/cuda/include',
            ],
            libraries=['cuda'],
            library_dirs=['/usr/local/cuda/lib64'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)