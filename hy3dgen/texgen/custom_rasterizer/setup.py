from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define your target GPU architectures
arch_flags = [
    '-gencode', 'arch=compute_90,code=sm_90',  # For NVIDIA H100 or later
    '-gencode', 'arch=compute_89,code=sm_89',  # For NVIDIA A30
    '-gencode', 'arch=compute_86,code=sm_86',  # For NVIDIA A100
    '-gencode', 'arch=compute_80,code=sm_80',  # For NVIDIA A100 with Tensor Cores
    '-gencode', 'arch=compute_75,code=sm_75',  # For NVIDIA T4
    '-gencode', 'arch=compute_70,code=sm_70',  # For NVIDIA V100
    '-gencode', 'arch=compute_61,code=sm_61',  # For older GPUs like GTX 10xx
    '-gencode', 'arch=compute_52,code=sm_52',  # Optional for older GPUs like GTX 9xx
]

custom_rasterizer_module = CUDAExtension(
    'custom_rasterizer_kernel',
    [
        'lib/custom_rasterizer_kernel/rasterizer.cpp',
        'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
        'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],  # Optimization for C++
        'nvcc': ['-O3'] + arch_flags  # Optimization for CUDA and architecture flags
    }
)

setup(
    packages=find_packages(),
    version='0.1',
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
