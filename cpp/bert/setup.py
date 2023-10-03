from glob import glob
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.getenv('CUDA_HOME') if os.getenv('CUDA_HOME')!=None else "/usr/local/cuda"

extra_compile_args = {"cxx": []}
extra_compile_args["nvcc"] = [
            # "-g",
            # "-G",
            # "-O0",
            "-O3",
            "-lcuda",
            "-lcudart",
            "-lcudart_static",
            "-DCUDA_HAS_FP16=1",
            "-DUSE_FP16=ON",
            "-DCUDA_ARCH_BIN=80",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-L/usr/local/cuda/targets/x86_64-linux/lib/stubs",
            "-L/usr/local/cuda/targets/x86_64-linux/lib/"
            ]
setup(name='bert_binding',
      ext_modules=[
        cpp_extension.CUDAExtension(
            'bert_binding', 
            ['bert_binding.cu'],
            include_dirs=[CUDA_HOME+"/include", this_dir+"/../"],
            # library_dirs=[CUDA_HOME+"/lib64"],
            library_dirs=[CUDA_HOME+"/lib64", CUDA_HOME+"/targets/x86_64-linux/lib/stubs", CUDA_HOME+"/targets/x86_64-linux"],
            extra_compile_args=extra_compile_args,
            libraries=["cuda", "cudart", "cudart_static"]
          ),
          cpp_extension.CUDAExtension(
            'souffle_bert_base', 
            ['bert_base_impl.cu'],
            include_dirs=[CUDA_HOME+"/include", this_dir+"/../"],
            # library_dirs=[CUDA_HOME+"/lib64"],
            library_dirs=[CUDA_HOME+"/lib64", CUDA_HOME+"/targets/x86_64-linux/lib/stubs", CUDA_HOME+"/targets/x86_64-linux"],
            extra_compile_args=extra_compile_args,
            libraries=["cuda", "cudart", "cudart_static"]
          )
        ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
