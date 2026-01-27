from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# setup(
#     name='cuda_reduce',
#     version='0.1.0',
#     author='Your Name',
#     description='CUDA reduce sum with PyTorch extension',
#     ext_modules=[
#         CUDAExtension(
#             name='cuda_reduce',
#             sources=[
#                 'pyblind.cpp',
#                 'kernel/reduce_sum.cu',
#             ],
#             extra_compile_args={
#                 'cxx': ['-O3', '-std=c++14'],
#                 'nvcc': [
#                     '-DNDEBUG',
#                     '-O3',
#                     '-g',
#                     '-lineinfo',
#                     '--use_fast_math',
#                     '-gencode=arch=compute_80,code=sm_80',
#                     '-gencode=arch=compute_90,code=sm_90',
#                 ]
#             },
#             include_dirs=[
#                 'kernel',
#                 '/usr/local/cuda/include',
#             ],
#             libraries=['cuda'],
#             library_dirs=['/usr/local/cuda/lib64'],
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     },
#     python_requires='>=3.6',
# )

setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(
            name='cuda_reduce',
            sources=[
                'pyblind.cpp',
                'kernel/reduction/reduce_sum.cu',
                'kernel/batch_reduction/batch_reduce.cu',
                'kernel/quant/per_token_quant_fp8.cu',
                'kernel/softmax/online_softmax.cu',
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
