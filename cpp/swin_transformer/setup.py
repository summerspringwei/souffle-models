from glob import glob
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.getenv('CUDA_HOME') if os.getenv('CUDA_HOME')!=None else "/usr/local/cuda"

extra_compile_args = {"cxx": ["-g", "-O0"]}
extra_compile_args["nvcc"] = [
            "-g",
            "-G",
            "-O0",
            # "-O3",
            "-DCUDA_HAS_FP16=1",
            "-DUSE_FP16=ON",
            "-DCUDA_ARCH_BIN=80",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            ]
setup(name='swin_transformer_binding',
      ext_modules=[
        cpp_extension.CUDAExtension(
          'swin_transformer_binding', 
          ['swin_transformer_binding.cu'],
          include_dirs=[CUDA_HOME+"/include", this_dir+"/../"],
          library_dirs=[CUDA_HOME+"/lib64"],
          extra_compile_args=extra_compile_args
          ),
        cpp_extension.CUDAExtension(
            'swin_trans_fc2', 
            ['swin_trans_fc2_binding.cu'],
            include_dirs=[CUDA_HOME+"/include", this_dir+"/../"],
            # library_dirs=[CUDA_HOME+"/lib64"],
            library_dirs=[CUDA_HOME+"/lib64", CUDA_HOME+"/targets/x86_64-linux/lib/stubs", CUDA_HOME+"/targets/x86_64-linux"],
            extra_compile_args=extra_compile_args,
            libraries=["cuda", "cudart", "cudart_static"]
          )
        ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
